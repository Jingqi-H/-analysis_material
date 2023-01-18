import matplotlib.pyplot as plt
import numpy as np

plt.figure()
N = 7
# 包含每个柱子对应值的序列
x = np.array(['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'])
# 绘制柱状图的均值和方差
means_men = np.array([53.38, 56.83, 59.60, 63.99, 64.28, 67.58, 72.43])
std_men = np.array([1.38, 0.78, 0.60, 3.41, 0.86, 1.42, 1.75])

# 包含每个柱子下标的序列
index = np.arange(N)
error_config=dict(elinewidth=4,ecolor='coral',capsize=6)
plt.rcParams['font.family'] = "Times New Roman"

# 柱子的宽度
width = 0.78
# 绘制柱状图
p2 = plt.bar(index, means_men, width,
                alpha=1, color='#3498db',align='center',
                yerr=std_men,error_kw=error_config,
                label='AUC')

# 添加标题
# plt.title('Monthly average rainfall')

# y轴刻度范围
plt.ylim((50, 78))

# 添加横坐标、纵横轴的刻度
plt.xticks(index, x, fontsize=15, rotation=30)
plt.yticks(np.arange(50, 80, 5), fontsize=15)  # 纵坐标刻度是5

# 添加图例
plt.legend(loc="upper left", fontsize=15)
# 添加网格线
plt.grid(axis="y", linestyle='-.')

# 保存
# plt.savefig('../PosterVideo/poster_material/sigf_auc.svg', format="svg",
#                 bbox_inches='tight', pad_inches=0.1)
plt.show()