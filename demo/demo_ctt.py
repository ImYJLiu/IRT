# coding=utf-8
# 经典测量理论
from __future__ import print_function, division, unicode_literals
from psy.ctt import BivariateCtt
from psy import data
import numpy as np

score = data['hdu_[2018-11-22 20.45.48, 2018-11-29 11.22.08]_11_R_matrix.dat']
# score = data['lsat.dat']
ctt = BivariateCtt(score)
print(ctt.get_alpha_reliability())
print(ctt.get_composite_reliability())
print(ctt.get_discrimination())
print(ctt.get_difficulty())
