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
import market as mkt

class DoubleBattery(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        
        self.MAX_ACTION = np.append(MAX_POWER, MAX_P_VPP)
        self.MAX_ACTION = np.append(self.MAX_ACTION, MAX_PRICE)
        
        # Actions: Ch/Disch Batt1, Ch/Disch Batt2, P_VPP, price
        high = np.array([1.0,1.0, 1.0, 1.0])
        low = np.array([-1.0,-1.0, -1.0, -1.0])
        self.action_space = spaces.Box(low=low, high=high, shape=(4,), dtype=np.float32)

        #high = np.array([MAX_SOE[0],MAX_SOE[1],MAX_LOAD[0]])   
        #low = np.array([MIN_SOE[0],MIN_SOE[1],MIN_LOAD[0]])
        # State Vector: SOE1,SOE2, L, P
        high = np.array([1.0,1.0,1.0,1.0])
        low = np.array([0.0,0.0,0.0,0.0])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        
        action = action * self.MAX_ACTION
        action = np.clip(action, -self.MAX_ACTION, self.MAX_ACTION)
        self.last_action = action
        battery_powers = action[0:1]
        bid = action[2]
        bid_price = action[3]
        
        self.dispatch, self.MCP = mkt.bid(RANDOM_BIDS, bid, bid_price, self.step_number)
        
        soe = np.array([self.state[0],self.state[1]]) * MAX_SOE
        load = self.state[2] * MAX_LOAD
        generation = self.state[3] * MAX_GEN
        self.last_load = load
        self.last_gen = generation
        soe_ = soe - battery_powers
        
        
        
        profit = self.dispatch * self.MCP
        # Cost Calculation
        obj = np.sum(PRICES * battery_powers)
        power_balance = HIGH_EPS * (np.sum(battery_powers) + generation - load - self.dispatch) ** 2
        SOE_constraint = np.sum(VHIGH_EPS * (MIN_SOE - np.minimum(MIN_SOE, soe_) + np.maximum(MAX_SOE, soe_) - MAX_SOE))
        
        costs = obj + power_balance + SOE_constraint - profit
        reward_norm = -costs/self.reward_basis
        self.imbalance = abs(np.sum(action) + generation - self.dispatch - load)
        
        if self.step_number < NUMBER_HOURS - 1:
            self.step_number = self.step_number + 1
            load = self.load_daily[self.step_number]
            generation = self.generation_daily[self.step_number]
            
            done = False
        else:
            done = True
            
        soe = soe_/MAX_SOE
        self.state = np.array([soe[0],soe[1],load,generation],dtype=np.float32)
        self.last_reward = -costs
        return self._get_obs(), reward_norm, done, {}
    
    def _get_obs(self):
        observation = self.state
        return observation

    def reset(self, mode = LEARN_MODE):
        
        self.step_number = 0
        if mode == LEARN_MODE:
            print("Learn Mode")
            self.load_daily = np.random.rand(NUMBER_HOURS)
            self.generation_daily = np.random.rand(NUMBER_HOURS)
            soe_init = np.maximum(np.random.rand(2),0.1)
            soe_init = soe_init.tolist()
        else:
            print("Test Mode")
            self.load_daily = LOAD_DAILY
            self.generation_daily = GENERATION_DAILY
            soe_init = SOE_init
        
        load_init = self.load_daily[0]
        gen_init = self.generation_daily[0]
        
        #Creation of State Array
        state = soe_init
        state.extend([load_init])
        state.extend([gen_init])
        self.state = np.array(state, dtype=np.float32)
        
        self.reward_basis = np.sum(PRICES * MAX_POWER) + HIGH_EPS * (np.sum(MAX_POWER) + MAX_GEN - MIN_LOAD) ** 2 + VHIGH_EPS * np.sum(MAX_POWER) + MAX_P_VPP * MAX_PRICE
        self.last_action = None
        self.last_load = None
        self.last_gen = None
        
        return self._get_obs()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.step_number}')
        print(f'Cost: {self.last_reward}')
    
    
    def retrieve_data(self):
        return self.call_reward(), self.call_imbalance(), self.call_load(), self.call_soe(),self.call_generation(), self.call_action(), self.call_MPC(), self.call_dispatch()
    
    def call_imbalance(self):
        return self.imbalance
    
    def call_load(self):
        return self.last_load
    
    def call_soe(self):
        return np.array([self.state[0], self.state[1]])*MAX_SOE
    
    def call_action(self):
        return self.last_action
    
    def call_reward(self):
        return self.last_reward
    
    def call_generation(self):
        return self.last_gen
    
    def call_step_number(self):
        return self.step_number
    
    def call_MPC(self):
        return self.MCP
    
    def call_dispatch(self):
        return self.dispatch

class SimpleBidder(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        
        self.MAX_ACTION = np.array([MAX_P_VPP, MAX_PRICE])

        
        # Actions: P_VPP, price
        high = np.array([1.0,1.0])
        low = np.array([0,0])
        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)

        # State Vector: Available power
        high = np.array([1.0])
        low = np.array([0.0])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        
        action = action * self.MAX_ACTION
        action = np.clip(action, 0, self.MAX_ACTION)
        self.last_action = action
        bid = action[0]
        bid_price = action[1]
        
        dispatch, MCP = mkt.bid(RANDOM_BIDS, bid, bid_price, self.step_number)
        self.dispatch = dispatch
        self.MCP = MCP
        
        available_power = self.state[0] * MAX_GEN
        self.last_gen = available_power    
        
        
        profit = self.dispatch * self.MCP
        # Cost Calculation
        if (bid > available_power) | (bid < - available_power) :
            imbalance = VHIGH_EPS #* abs(bid-available_power)
            profit = 0
        else:
            imbalance = 0
            profit = self.dispatch * self.MCP
        
        costs = imbalance - profit
        reward_norm = -costs/self.reward_basis
        self.imbalance = imbalance
        
        if self.step_number < NUMBER_HOURS - 1:
            self.step_number = self.step_number + 1

            available_power = self.generation_daily[self.step_number]
            
            done = False
        else:
            done = True
            
        
        self.state = np.array([available_power],dtype=np.float32)
        self.last_reward = -costs
        return self._get_obs(), reward_norm, done, {}
    
    def _get_obs(self):
        observation = self.state
        return observation

    def reset(self, mode = LEARN_MODE):
        
        self.step_number = 0
        if mode == LEARN_MODE:
            print("Learn Mode")
            
            self.generation_daily = np.random.rand(NUMBER_HOURS)

        else:
            print("Test Mode")
            self.load_daily = LOAD_DAILY
            self.generation_daily = GENERATION_DAILY
            soe_init = SOE_init
        
        gen_init = self.generation_daily[0]
        
        #Creation of State Array

        self.state = np.array([gen_init], dtype=np.float32)
        
        self.reward_basis = VHIGH_EPS * MAX_P_VPP + MAX_P_VPP * MAX_PRICE
        self.last_action = None

        self.last_gen = None
        
        return self._get_obs()
    
    def retrieve_data(self):
        return self.call_reward(), self.call_imbalance(),self.call_generation(), self.call_action(), self.call_MPC(), self.call_dispatch()
    
    def call_imbalance(self):
        return self.imbalance
    
    
    def call_action(self):
        return self.last_action
    
    def call_reward(self):
        return self.last_reward
    
    def call_generation(self):
        return self.last_gen
    
    def call_step_number(self):
        return self.step_number
    
    def call_MPC(self):
        return self.MCP
    
    def call_dispatch(self):
        return self.dispatch