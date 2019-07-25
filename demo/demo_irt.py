# coding=utf-8
# 单维IRT参数估计
from __future__ import print_function, division, unicode_literals
from psy import Irt, data
import matplotlib.pyplot as plt
import numpy as np
from psy.settings import X_WEIGHTS, X_NODES
score = data['hdu_[2018-11-22 20.45.48, 2018-11-29 11.22.08]_11_R_matrix.dat']
score1 = data['hdu_[2018-11-22 20.45.48, 2018-11-29 11.22.08]_only11_R_matrix.dat']

model = Irt(scores=score, link='logit')
res = model.fit()

# 绘制第一个题的散点图
Y_NODES=1/(1+np.exp(-(res[0][0]*X_NODES+res[1][0])))
plt.scatter(X_NODES,Y_NODES,alpha=0.6)
plt.title('ICC curve')
plt.xlabel('ability  parameter')
plt.ylabel('probability of right')
plt.show()


# plt.plot(range(len(a)), a, marker='o', mec='r', mfc='w',label=u'a参数曲线图')
# plt.plot(range(len(b)), b, marker='*', ms=10,label=u'b参数曲线图')
# plt.legend()  # 让图例生效
# plt.margins(0)
#
# plt.show()
#
# #绘制折线图
# predict=1/(1+np.exp(-(res[0]*X_NODES+res[1])))
# rel=score1
# RMSE=np.sqrt(np.mean(predict-rel)**2)
# MAE=np.mean(np.abs(predict-rel))
# print('rmse {:.4f}  mae {:.4f} '.format(RMSE,MAE))
#
