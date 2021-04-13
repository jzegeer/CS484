## CS 484 - Project
## March 2021
## Anh Nguyen & Jake Zegeer

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

## Read train data to a table
train_data = pd.read_csv('train.csv')

## Read test data to a table
test_data = pd.read_csv('test.csv')

## To view: Right Click -> Run Current File in Interactive Window
## Display first five rows and their 81 columns
print(train_data.head())

#Creates a main target value 
print(train_data.groupby('target').size())




















