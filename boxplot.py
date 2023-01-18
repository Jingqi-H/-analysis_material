import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
