import os

import torch
from sklearn import metrics
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from keras.utils.np_utils import to_categorical


def calculate_roc(y_score, y_test, n_classes):
    """
    https://blog.csdn.net/liujh845633242/article/details/102938143

    根据sklearn参考文档
    y_test是二值，y_score是概率
    :param y_score:是得到预测结果，他是概率值，并且是array
    :param y_test:是gt
    :param save_results: 保存路径
    :return:
    """
    if n_classes == 2:
        y_test = to_categorical(y_test, n_classes)
    else:
        # label_binarize对于两个以上的分类，可以将1维转化为多维，对于二分类，就还是一维,classes>=3才能成功使用:
        y_test = label_binarize(y_test, classes=list(range(n_classes)))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc


def cm_metric(gt_labels, pre_prob, cls_num=1):
    """

    :param gt_labels:
    :param pre_prob: 模型预测的概率值，在cpu
    :param cls_num: 预测的类别数
    :return:
    """
    if cls_num == 1:
        pre_label = pre_prob > 0.5
        cnf_matrix = confusion_matrix(gt_labels, pre_label, labels=None, sample_weight=None)
        Accary = (cnf_matrix[1, 1] + cnf_matrix[0, 0]) / (
                cnf_matrix[1, 1] + cnf_matrix[0, 1] + cnf_matrix[0, 0] + cnf_matrix[1, 0])
        Recall = cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[1, 0])
        Precision = cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[0, 1])
        Specificity = cnf_matrix[0, 0] / (cnf_matrix[0, 1] + cnf_matrix[0, 0])

        fpr, tpr, thresholds = metrics.roc_curve(gt_labels, pre_prob)
        roc_auc = metrics.auc(fpr, tpr)
    else:
        # pre_label = torch.argmax(pre_prob, dim=1)
        pre_label = np.argmax(np.array(pre_prob), axis=1)
        cnf_matrix = confusion_matrix(gt_labels, pre_label, labels=None, sample_weight=None)

        sum_TP = 0
        for i in range(cls_num):
            sum_TP += cnf_matrix[i, i]
        Accary = sum_TP / np.sum(cnf_matrix)
        Acc_all, Recall_all, Precision_all, Specificity_all, auc_all = [], [], [], [], []
        for i in range(cls_num):
            TP = cnf_matrix[i, i]
            FP = np.sum(cnf_matrix[i, :]) - TP
            FN = np.sum(cnf_matrix[:, i]) - TP
            TN = np.sum(cnf_matrix) - TP - FP - FN
            precision = (TP / (TP + FP)) if TP + FP != 0 else 0.
            recall = (TP / (TP + FN)) if TP + FN != 0 else 0.
            specificity = (TN / (TN + FP)) if TN + FP != 0 else 0.
            Recall_all.append(recall)
            Precision_all.append(precision)
            Specificity_all.append(specificity)

        fpr, tpr, roc_auc = calculate_roc(pre_prob, gt_labels, cls_num)
        # micro：多分类；macro：计算二分类metrics的均值，为每个类给出相同权重的分值。
        Recall, Precision, Specificity, roc_auc = np.around(np.mean(Recall_all), 4), \
                                                  np.around(np.mean(Precision_all), 4), \
                                                  np.around(np.mean(Specificity_all), 4), \
                                                  np.around(roc_auc["macro"], 4)

    return Accary, Recall, Precision, Specificity, roc_auc


if __name__ == '__main__':
    import pandas as pd
    from pathlib import Path

    gt = torch.tensor([1,1,1,1,0,0,0,1,0,1])
    prob = torch.randn([10, 2])
    score = cm_metric(gt, prob, cls_num=prob.shape[1])
    print(score)
