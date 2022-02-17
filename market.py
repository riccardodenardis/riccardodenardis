# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:38:42 2022

@author: rdnar
"""

import gurobipy as gb
import numpy as np
import pandas as pd


class expando(object):
    '''   
        A small class which can have attributes set
    '''
    pass

class market:
    
    def __init__(self, SUPPLIES, DEMANDS, SUPPLY_PRICES, DEMAND_PRICES, SUPPLY_BIDS, DEMAND_BIDS, HOURS):
        self.variables = expando()
        self.constraints = expando()
        self._build_model(SUPPLIES, DEMANDS, SUPPLY_PRICES, DEMAND_PRICES, SUPPLY_BIDS, DEMAND_BIDS, HOURS)
        
    def optimize(self):
        self.model.optimize()
    
    def _build_model(self, SUPPLIES, DEMANDS, SUPPLY_PRICES, DEMAND_PRICES, SUPPLY_BIDS, DEMAND_BIDS, HOURS):
        self.model = gb.Model()
        self.model.Params.OutputFlag = 0
        self.model.Params.LogToConsole = 0
        self.model.Params.TimeLimit = 100
        self.model.Params.QCPDual = 1
        
        self._build_variables(SUPPLIES, DEMANDS, SUPPLY_BIDS, DEMAND_BIDS, HOURS)
        self._build_objective(SUPPLIES, DEMANDS, SUPPLY_PRICES, DEMAND_PRICES, HOURS)
        self._build_constraints(SUPPLIES, DEMANDS,  HOURS)
        
    def _build_variables(self, SUPPLIES, DEMANDS, SUPPLY_BIDS, DEMAND_BIDS, HOURS):
        m = self.model
        
        # Demands
        self.variables.demand = {}
        
        for d in range(len(DEMANDS)):
            for h in range(len(HOURS)):
                self.variables.demand[h,d] = m.addVar(lb = 0, ub = DEMAND_BIDS[h,d], name = 'Demand of load {0} at time {1}'.format(d,h))
        
        # Supplies
        self.variables.supply = {}
        
        for s in range(len(SUPPLIES)):
            for h in range(len(HOURS)):
                self.variables.supply[h,s] = m.addVar(lb = 0, ub = SUPPLY_BIDS[h,s], name = 'Supply of generator {0} at time {1}'.format(s,h))
                
    def _build_objective(self, SUPPLIES, DEMANDS, SUPPLY_PRICES, DEMAND_PRICES, HOURS):
        
        m = self.model
        m.setObjective(gb.quicksum(DEMAND_PRICES[h,d] * self.variables.demand[h,d] for d in range(len(DEMANDS)) for h in range(len(HOURS))) - gb.quicksum(SUPPLY_PRICES[h,s]* self.variables.supply[h,s] for s in range(len(SUPPLIES)) for h in range(len(HOURS))), gb.GRB.MAXIMIZE)
        
    def _build_constraints(self, SUPPLIES, DEMANDS, HOURS):
        m = self.model 
        
        self.constraints.powerbalance = {}
        
        for h in range(len(HOURS)):
            self.constraints.powerbalance[h] = m.addConstr(gb.quicksum(self.variables.supply[h,s] for s in range(len(SUPPLIES))) - gb.quicksum(self.variables.demand[h,d] for d in range(len(DEMANDS))), gb.GRB.EQUAL, 0, name = "Power balance at time {0}".format(h))
    
    def get_results(self, HOURS, SUPPLIES, DEMANDS):
        
        supplies_res = np.zeros((len(HOURS), len(SUPPLIES)))
        demands_res = np.zeros((len(HOURS), len(DEMANDS)))
        MCP = np.zeros(len(HOURS))

        for h in range(len(HOURS)):
            for d in range(len(DEMANDS)):
                demands_res[h,d] = self.variables.demand[h,d].x
                
            for s in range(len(SUPPLIES)):
                supplies_res[h,s] = self.variables.supply[h,s].x
                

            MCP[h] = - self.constraints.powerbalance[h].Pi
        
        return demands_res, supplies_res, MCP

def bid(RANDOM_BIDS, bid, bid_price, start_hour, stddev_sb = 10, stddev_db = 20, stddev_sp = 0.2, stddev_dp = 0 ):
    SUPPLY_BIDS, DEMAND_BIDS, SUPPLY_PRICES, DEMAND_PRICES, SUPPLIES, DEMANDS, HOURS = read_data()
    
    if RANDOM_BIDS:
        SUPPLY_PRICES, DEMAND_PRICES, SUPPLY_BIDS, DEMAND_BIDS = randomize_bids(SUPPLIES, DEMANDS, SUPPLY_PRICES, DEMAND_PRICES, SUPPLY_BIDS, DEMAND_BIDS, HOURS, stddev_sb, stddev_db, stddev_sp, stddev_dp)
    
    HOURS_ = HOURS[start_hour:start_hour+bid.size]
    
    SUPPLY_PRICES = SUPPLY_PRICES[HOURS_,:]
    DEMAND_PRICES = DEMAND_PRICES[HOURS_,:]
    SUPPLY_BIDS = SUPPLY_BIDS[HOURS_,:]
    DEMAND_BIDS = DEMAND_BIDS[HOURS_,:]
    
    if bid >= 0:
        SUPPLY_PRICES = np.append(SUPPLY_PRICES,bid_price)
        SUPPLY_BIDS = np.append(SUPPLY_BIDS,bid)
        SUPPLIES = SUPPLIES.append(pd.Index(["VPP"]))
    else:
        DEMAND_PRICES = np.append(DEMAND_PRICES,bid_price)
        DEMAND_BIDS = np.append(DEMAND_BIDS,abs(bid))
        DEMANDS = DEMANDS.append(pd.Index(["VPP"]))
    
    SUPPLY_PRICES = SUPPLY_PRICES.reshape(len(HOURS_), SUPPLIES.size)
    SUPPLY_BIDS = SUPPLY_BIDS.reshape(len(HOURS_), SUPPLIES.size)

    DEMAND_PRICES = DEMAND_PRICES.reshape(len(HOURS_), DEMANDS.size)
    DEMAND_BIDS = DEMAND_BIDS.reshape(len(HOURS_), DEMANDS.size)

    
    m = market(SUPPLIES, DEMANDS, SUPPLY_PRICES, DEMAND_PRICES, SUPPLY_BIDS, DEMAND_BIDS, HOURS_)
    m.optimize()
    
    demands_res, supplies_res, MCP = m.get_results(HOURS_, SUPPLIES, DEMANDS)
    
    if bid >=0:
        return supplies_res[:,-1][0], MCP[0]
    else:
        return -demands_res[:,-1][0], MCP[0]
    
    
def randomize_bids(SUPPLIES, DEMANDS, SUPPLY_PRICES, DEMAND_PRICES, SUPPLY_BIDS, DEMAND_BIDS, HOURS, stddev_sb, stddev_db, stddev_sp, stddev_dp):
    
    for h in HOURS:
        for s in range(len(SUPPLIES)):
            noise = np.random.normal(0,stddev_sp,1)
            SUPPLY_PRICES[h,s] = SUPPLY_PRICES[h,s] + noise
            noise = np.random.normal(0,stddev_sb,1)
            SUPPLY_BIDS[h,s] = SUPPLY_BIDS[h,s] + noise
            
        for d in range(len(DEMANDS)):
            noise = np.random.normal(0,stddev_dp,1)
            DEMAND_PRICES[h,d] = DEMAND_PRICES[h,d] + noise
            noise = np.random.normal(0,stddev_db,1)
            DEMAND_BIDS[h,d] = DEMAND_BIDS[h,d] + noise
    
    return SUPPLY_PRICES, DEMAND_PRICES, SUPPLY_BIDS, DEMAND_BIDS

def read_data():
    supply_bids = pd.read_csv("data/supply_bids.csv", delimiter = ",")
    demand_bids = pd.read_csv("data/demand_bids.csv", delimiter = ",")
    supply_prices = pd.read_csv("data/supply_prices.csv", delimiter = ",")
    demand_prices = pd.read_csv("data/demand_prices.csv", delimiter = ",")
    
    HOURS = supply_bids["HOURS"].to_numpy()
    
    
    supply_bids = supply_bids.set_index("HOURS")
    demand_bids = demand_bids.set_index("HOURS")
    supply_prices = supply_prices.set_index("HOURS")
    demand_prices = demand_prices.set_index("HOURS")
    
    SUPPLIES = supply_bids.columns
    DEMANDS = demand_bids.columns
    
    supply_bids = supply_bids.to_numpy()
    demand_bids = demand_bids.to_numpy()
    supply_prices = supply_prices.to_numpy()
    demand_prices = demand_prices.to_numpy()
    
    return supply_bids, demand_bids, supply_prices, demand_prices, SUPPLIES, DEMANDS, HOURS

