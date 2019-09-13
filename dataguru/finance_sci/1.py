# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:55:37 2018

@author: DELL
"""

import pandas as pd
import os
import ffn

price = [1.0, 1.01, 1.05, 1.1, 1.11, 1.07, 1.03, 1.03, 1.01, 1.02,1.04, 1.05, 1.07, 1.06,1.05, 1.06, 1.07, 1.09, 1.12, 1.18, 1.15, 1.15, 1.18, 1.16, 1.19, 1.17, 1.17, 1.18,1.19, 1.23]

data=pd.DataFrame(price)

returnS=ffn.to_returns(data).dropna()
max_dropdown=ffn.calc_max_drawdown((1+returnS).cumprod())

print ("最大回撤是: %.4f" %max_dropdown)