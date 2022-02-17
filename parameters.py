# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:27:28 2022
# PARAMETERS
@author: rdnar
"""
import numpy as np

LEARN_MODE = 0
TEST_MODE = 1

NUMBER_HOURS = 24
PRICES = np.array([2.0,1.0])
HIGH_EPS = 100000
VHIGH_EPS = 1000

MAX_POWER = np.array([100.0,100.0])
#MIN_POWER = np.array([0.0,0.0])
MAX_LOAD = 100.0
MIN_LOAD = 0.0
MAX_GEN = 100.0
MAX_SOE = np.array([2500.0,2500.0])
MIN_SOE = np.array([0.0,0.0])

MAX_P_VPP = 100
MAX_PRICE = 4

RANDOM_BIDS = False

#Testing Scenario
SOE_init = ([1000,1000]/MAX_SOE).tolist()
#GENERATION_DAILY = np.ones(24)
GENERATION_DAILY = np.array([0,0,0,0,0,0,50,50,50,100,100,100,50,50,50,100,100,100,70,70,70,0,0,0])/MAX_GEN
LOAD_DAILY = np.array([0,0,0,0,0,0,60,60,60,100,100,100,60,60,60,70,70,100,50,90,70,10,0,0])/MAX_LOAD