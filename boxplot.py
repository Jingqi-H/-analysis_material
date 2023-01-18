import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# x = np.array(['VGG', 'GoogLeNet', 'ResNet', 'FixMatch', 'CCT', 'UPS', 'VTFN'])

data = {
'AAA': [55.24,51.52,49.5,50,47.79],
'BBB': [57.9,56.34,54.62,57.93,57.37],
'CCC': [59,60.19,62.5,63.26,60.19],
"DDD": [63.41,69.26,59.73,63.55,61.43],
"EEE": [63.42,65.14,61.28,62.86,58.31],
"FFF": [69,66.16,57,66,74.09],
"GGG": [74.67,74.92,75.49,75.84,78.25]
}
df = pd.DataFrame(data)
# notch 改变箱子形状，showmeans显示均值（实心三角），sym显示异常值
df.plot.box(showmeans=True)
plt.grid(linestyle="--", alpha=0.3)


index = np.arange(7)
plt.xticks(fontsize=18, rotation=20)
plt.yticks(fontsize=20)  # 纵坐标刻度是5

plt.tight_layout()
# 保存
# plt.savefig('../figures/sigf_auc.pdf', format="pdf",
#                 bbox_inches='tight', pad_inches=0)
plt.show()





















# plt.figure()
# N = 8
# # 包含每个柱子对应值的序列
# x = np.array(['VGG', 'GoogLeNet', 'ResNet', 'FixMatch', 'CCT', 'UPS', 'GCNET', 'VTFN'])
# # 绘制柱状图的均值和方差
# means_men = np.array([53.38, 56.83, 59.60, 63.99, 64.28, 67.58, 72.43, 75.23])
# std_men = np.array([1.38, 0.78, 0.60, 3.41, 0.86, 1.42, 1.75, 0.53])
#
# # 包含每个柱子下标的序列
# index = np.arange(N)
# error_config=dict(elinewidth=4,ecolor='coral',capsize=6)
# plt.rcParams['font.family'] = "Times New Roman"
#
# # 柱子的宽度
# width = 0.78
# # 绘制柱状图
# p2 = plt.bar(index, means_men, width,
#                 alpha=1, color='#3498db',align='center',
#                 yerr=std_men,error_kw=error_config,
#                 label='AUC')
#
# # 添加标题
# # plt.title('Monthly average rainfall')
#
# # y轴刻度范围
# plt.ylim((50, 78))
#
# # 添加横坐标、纵横轴的刻度
# plt.xticks(index, x, fontsize=15, rotation=30)
# plt.yticks(np.arange(50, 80, 5), fontsize=15)  # 纵坐标刻度是5
#
# # 添加图例
# plt.legend(loc="upper left", fontsize=15)
# # 添加网格线
# plt.grid(axis="y", linestyle='-.')
#
# # 保存
# # plt.savefig('../vtfn_sigf_auc.png', format="png",
# #                 bbox_inches='tight', pad_inches=0.1)
# plt.show()