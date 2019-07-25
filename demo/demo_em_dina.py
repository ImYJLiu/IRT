# coding=utf-8
# 认知诊断DINA模型的EM参数估计
from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import EmDina, MlDina
from psy.utils import r4beta

# binomial二项分布
#attrs是Q矩阵
attrs = np.random.binomial(1, 0.5, (5, 60))
# skills是学生做题情况 shape=[做题记录*题目]
skills = np.random.binomial(1, 0.7, (1000, 5))
skills_predict=np.random.binomial(1,0.7,(1000,5))

# 4参数beta分布
# (疑问?博客说是每一个题都有g,s,这里是每一个知识点都有g,s)
g = r4beta(1, 2, 0, 0.6, (1, 60))
no_s = r4beta(2, 1, 0.4, 1, (1, 60))

# yita理论做题情况          shape=[做题记录数*知识点数]
# p_val理论情况下作对的概率 shape=[做题记录数*知识点数]
# score理论情况下的做题情况 shape=[做题记录数*知识点数]
temp = EmDina(attrs=attrs)
yita = temp.get_yita(skills)
p_val = temp.get_p(yita, guess=g, no_slip=no_s)
score = np.random.binomial(1, p_val)

# 估计项目参数
em_dina = EmDina(attrs=attrs, score=score)
est_no_s, est_g = em_dina.em()

# print(np.mean(np.abs(est_no_s - no_s)))
# print(np.mean(np.abs(est_g - g)))

# 估计被试掌握技能情况
dina_est = MlDina(guess=est_g, no_slip=est_no_s, attrs=attrs, score=score)
est_skills = dina_est.solve()

print('MAE',np.mean(np.abs(skills_predict-est_skills)))
print("RMSE",np.sqrt(np.mean(est_skills-skills_predict)**2))
