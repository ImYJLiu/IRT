#conding=UTF-8
from psy import data
import numpy as np
import pandas as pd
from psy.utils import r4beta
from psy import EmDina, MlDina

# 加载数据
# score = data['hdu_[2018-11-22 20.45.48, 2018-11-29 11.22.08]_pre11_0.3_R_matrix.dat']
# score_predict = data['hdu_[2018-11-22 20.45.48, 2018-11-29 11.22.08]_only12_0.3_R_matrix.dat']
# attrs = data['hdu_0.3_Q_Matrix.dat'].T

score = data['hdu_[2018-11-22 20.45.48, 2018-11-29 11.22.08]_only12_0.2_R_matrix.dat']
score_predict = data['hdu_[2018-11-22 20.45.48, 2018-11-29 11.22.08]_pre11_0.2_R_matrix.dat']
attrs = data['hdu_0.2_Q_Matrix.dat'].T

# 4参数beta分布
g = r4beta(1, 2, 0, 0.6, (1, 12))
no_s = r4beta(2, 1, 0.4, 1, (1, 12))

temp = EmDina(attrs=attrs)
yita = temp.get_yita(score)
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

print('MAE',np.mean(np.abs(score_predict-est_skills)))
print("RMSE",np.sqrt(np.mean(est_skills-score_predict)**2))
