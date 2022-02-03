# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 09:16:47 2022
# MyEnvironment
@author: rdnar
"""

import gym
from gym import spaces
import numpy as np
from gym.utils import seeding
from parameters import *


class DoubleBattery(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        
        # high = MAX_POWER
        # low = MIN_POWER
        high = np.array([1.0,1.0])
        low = np.array([0.0,0.0])
        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)

        #high = np.array([MAX_SOE[0],MAX_SOE[1],MAX_LOAD[0]])   
        #low = np.array([MIN_SOE[0],MIN_SOE[1],MIN_LOAD[0]])
        high = np.array([1.0,1.0,1.0])
        low = np.array([0.0,0.0,0.0])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        
        action = action * MAX_POWER
        action = np.clip(action, MIN_POWER, MAX_POWER)
        self.last_action = action
        
        soe = np.array([self.state[0],self.state[1]]) * SOE_init
        load = self.state[2] * MAX_LOAD
        self.last_load = load
        soe_ = soe - action
        
        costs = np.sum(PRICES * action) + HIGH_EPS * (np.sum(action) - load) ** 2
        reward_norm = -costs/self.reward_basis
        self.imbalance = (np.sum(action) - load)
        if self.step_number < NUMBER_HOURS - 1:
            self.step_number = self.step_number + 1
            load = self.load_daily[self.step_number]
            soe = soe_/SOE_init
            done = False
        else:
            done = True
        
        self.state = np.array([soe[0],soe[1],load],dtype=np.float32)
        self.last_reward = -costs
        return self._get_obs(), reward_norm, done, {}
    
    def _get_obs(self):
        observation = self.state
        return observation

    def reset(self):
        self.step_number = 0
        self.load_daily = np.random.rand(NUMBER_HOURS)
        self.reward_basis = np.sum(PRICES * MAX_POWER) + HIGH_EPS * (np.sum(MAX_POWER) - MIN_LOAD) ** 2
        load_init = self.load_daily[0] 
        
        self.state = np.array([max(np.random.rand(1)[0],0.1), max(np.random.rand(1)[0],0.1), load_init], dtype=np.float32)
        self.last_action = None
        self.last_load = None
        return self._get_obs()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.step_number}')
        print(f'Cost: {self.last_reward}')
    
    
    def retrieve_data(self):
        return self.call_reward(), self.call_imbalance(), self.call_load(), self.call_soe(), self.call_action()
    
    def call_imbalance(self):
        return self.imbalance
    
    def call_load(self):
        return self.last_load
    
    def call_soe(self):
        return np.array([self.state[0], self.state[1]])*SOE_init
    
    def call_action(self):
        return self.last_action
    
    def call_reward(self):
        return self.last_reward
    
