import matplotlib.pyplot as plt
import numpy as np

# A = [[28, 3, 5, 6, 3, 7, 2, 4, 9, 95],
#      [58, 6, 13, 13, 5, 12, 17, 11, 10, 102],
#      [60, 22, 21, 41, 11, 5, 1, 2, 3, 4]]

A = [[68.42, 71.43, 76.92, 32.37],
    [64.3, 68.7, 73.3, 31.8],
    [65.6, 67.6, 71.5, 30.9]]
# 生成横坐标
x_labels = []
# for item in range(0, 100, 10):
#    x = item + 10
#    if x == 10:
#        x_labels.append("{}~{}".format(0, 10))
#    else:
#        x_labels.append("{}".format(x))
x_labels = ['Cornell', 'Texas', 'Wisconsin', 'Film']
# 生成横坐标范围
x = np.arange(4)
# 生成多柱图
plt.bar(x + 0.00, A[0], color='orange', width=0.3, label="MOSL with all")
plt.bar(x + 0.30, A[1], color='royalblue', width=0.3, label="MOSL with decoder")
plt.bar(x + 0.60, A[2], color='green', width=0.3, label="MOSL with discriminator")
# 图片名称
# plt.title('多柱图', fontproperties=prop)
# 横坐标绑定
plt.xticks(x + 0.30, x_labels)

plt.ylabel("Classification Accuracy")
# 生成图片
plt.legend(loc="best")
plt.grid()
plt.savefig("z3.png", dpi=700)
# plt.show()