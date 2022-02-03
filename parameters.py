# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:27:28 2022
# PARAMETERS
@author: rdnar
"""
import numpy as np

NUMBER_HOURS = 24
PRICES = np.array([2.0,1.0])
HIGH_EPS = 10
MAX_POWER = np.array([100.0,100.0])
MIN_POWER = np.array([0.0,0.0])
MAX_LOAD = 10.0
MIN_LOAD = 0.0
MAX_SOE = np.array([2500.0,2500.0])
MIN_SOE = np.array([0.0,0.0])
SOE_init = 2500.0