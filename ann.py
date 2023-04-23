#importing libraries
import pandas as pd
import tensorflow as tf
import numpy as np


#preprocessing

data = pd.read_csv('Churn_Modelling.csv')
x = data.iloc[:, 3:-1].values #in the dataset there is no significance for other columns
y = data.iloc[:,-1].values #only the last column
print(x)