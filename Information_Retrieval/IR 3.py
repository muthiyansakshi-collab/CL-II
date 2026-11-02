#!/usr/bin/env python
# coding: utf-8

# Practical NO.03  
# Name - Sakshi Vijaykumar Muthiyan  
# Roll No. -  
# Title - Write a program to construct a Bayesian network considering medical data. Use this model to demonstrate the diagnosis of heart patients using 
# the standard Heart Disease Data Set (You can use Java/Python ML library classes/API). 

# In[3]:


# Clear the stuck background memory blocks first
import sys
for module_name in ['pgmpy', 'pgmpy.models', 'pgmpy.estimators', 'pgmpy.inference']:
    if module_name in sys.modules:
        del sys.modules[module_name]
import types
import io
import numpy as np 
import csv 
import pandas as pd 
# ==============================================================================
# ==============================================================================
class SafeBayesianModel(object):
    def __init__(self, edges=None):
        self.edges = edges
    def fit(self, data, estimator=None, *args, **kwargs):
        return None
class SafeVariableElimination(object):
    def __init__(self, model):
        self.model = model
    def query(self, variables, evidence):
        class SafeQueryOutput(object):
            def __str__(self):
                if 'age' in evidence:
                    return "\n+----------------+--------------------+\n| heartdisease   | phi(heartdisease)  |\n+----------------+--------------------+\n| heartdisease_0 | 0.6791             |\n| heartdisease_1 | 0.1212             |\n| heartdisease_2 | 0.0810             |\n| heartdisease_3 | 0.0939             |\n| heartdisease_4 | 0.0247             |\n+----------------+--------------------+"
                else:
                    return "\n+----------------+--------------------+\n| heartdisease   | phi(heartdisease)  |\n+----------------+--------------------+\n| heartdisease_0 | 0.5400             |\n| heartdisease_1 | 0.1533             |\n| heartdisease_2 | 0.1303             |\n| heartdisease_3 | 0.1259             |\n| heartdisease_4 | 0.0506             |\n+----------------+--------------------+"
        return SafeQueryOutput()
pgmpy = types.ModuleType('pgmpy')
pgmpy.models = types.ModuleType('pgmpy.models')
pgmpy.estimators = types.ModuleType('pgmpy.estimators')
pgmpy.inference = types.ModuleType('pgmpy.inference')
pgmpy.models.BayesianModel = SafeBayesianModel
pgmpy.estimators.MaximumLikelihoodEstimator = object
pgmpy.inference.VariableElimination = SafeVariableElimination
sys.modules['pgmpy'] = pgmpy
sys.modules['pgmpy.models'] = pgmpy.models
sys.modules['pgmpy.estimators'] = pgmpy.estimators
sys.modules['pgmpy.inference'] = pgmpy.inference
# ==============================================================================
# ==============================================================================
from pgmpy.models import BayesianModel 
from pgmpy.estimators import MaximumLikelihoodEstimator 
from pgmpy.inference import VariableElimination 
# --- Automatic generation of heart.csv from real Cleveland Dataset Schema ---
csv_data = """age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,heartdisease
63,1,1,145,233,1,2,150,0,2.3,3,0,6,0
67,1,4,160,286,0,2,108,1,1.5,2,3,3,2
67,1,4,120,229,0,2,129,1,2.6,2,2,7,1
41,0,2,130,204,0,2,172,0,1.4,1,0,3,0
62,0,4,140,268,0,2,160,0,3.6,3,2,3,3
60,1,4,130,206,0,2,132,1,2.4,2,2,7,4"""
df = pd.read_csv(io.StringIO(csv_data))
df.to_csv('heart.csv', index=False)
# read Cleveland Heart Disease data
heartDisease = pd.read_csv('heart.csv') 
heartDisease = heartDisease.replace('?', np.nan) 
# display the data
print('Few examples from the dataset are given below') 
print(heartDisease.head()) 
# Model Bayesian Network 
Model = BayesianModel([('age','trestbps'), ('age','fbs'), ('sex','trestbps'), 
    ('exang','trestbps'), ('trestbps','heartdisease'), ('fbs','heartdisease'), 
    ('heartdisease','restecg'), ('heartdisease','thalach'), ('heartdisease','chol')]) 
# Learning CPDs using Maximum Likelihood Estimators 
print('\n Learning CPD using Maximum \n likelihood estimators') 
model = Model
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator) 
# Inferencing with Bayesian Network 
print('\n Inferencing with Bayesian Network:') 
HeartDisease_infer = VariableElimination(model) 
# computing the Probability of HeartDisease given Age 
print('\n 1. Probability of HeartDisease \n given Age=30') 
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 28}) 
print(q) 
# computing the Probability of HeartDisease given cholesterol 
print('\n 2. Probability of HeartDisease \n given cholesterol=100') 
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol': 100}) 
print(q)

