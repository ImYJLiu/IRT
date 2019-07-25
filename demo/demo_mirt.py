# coding=utf-8
# 多维项目反应理论的参数估计
from __future__ import print_function, division, unicode_literals
from psy import Mirt, data
import numpy as np

score = data['hdu_[2018-11-22 20.45.48, 2018-11-29 11.22.08]_pre11_0.3_R_matrix.dat']
score1 = data['hdu_[2018-11-22 20.45.48, 2018-11-29 11.22.08]_only12_0.3_R_matrix.dat']
# score = data['lsat.dat']
mirt= Mirt(scores=score, dim_size=2)
res=mirt.em()

mirt= Mirt(scores=score1, dim_size=2)
res1=mirt.em()

MAE,MEAN=mirt.predict(res[0],res[1],res1[0],res1[1])
print(MAE,MEAN)

# Y_NODES=1/(1+np.exp(-(res[0][0]*X_NODES+res[1][0])))
# plt.scatter(X_NODES,Y_NODES,alpha=0.6)
# plt.title('ICC curve')
# plt.xlabel('ability  parameter')
# plt.ylabel('probability of right')
# plt.show()
