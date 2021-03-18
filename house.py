## CS 484 - Project
## March 2021
## Anh Nguyen & Jake Zegeer

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## Read train data to a table
train_data = pd.read_csv('train.csv')

## To view: Right Click -> Run Current File in Interactive Window
## Display first five rows and their 81 columns
train_data.head(5)