#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 00:06:28 2026

@author: shanice
"""
import numpy as np
#if we assume low risk is anything below 0.2 with possible variation of about 0.23 (can decide later just a baseline)
env_context={"mean risk":0.2, "stdv":0.15, "iterations":10000} 
def monte_carlo(sim_para):
    
    simulations=np.random.normal(
        sim_para["mean risk"],
        sim_para["stdv"],
        sim_para["iterations"]) #generating 10k random samples based on what we perceive as low risk
   
    prior_prob=np.mean(simulations) #prior probability 
    
    var_threshold=np.percentile(simulations, 95) #if it exceeds means high risk
    
    return prior_prob, var_threshold

prior_prob, var_95=monte_carlo(env_context)

print(f"P(T)={prior_prob:.4f}")
print(f"VaR Threshold={var_95:.4f}")

#base code
