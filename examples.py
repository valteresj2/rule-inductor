# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 13:56:08 2022

@author: valteresj
"""
import machine_inductor as ir
import pandas as pd



path='C:/Users/valteresj/Documents/Projetos/automl'




dt=pd.read_csv(path+'/bank-full.csv',sep=';')


train=ir.inductor(target='y',
n_bins=5,
label='yes',
level_rule=4,
perc=0.05,
t_lift=0.2,
dt=dt,
path=path,
name_project='teste1')

rules=train.inductor_machine()


pred=train.pred(new_data=dt,rules_select=rules[0:4])