import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('seaborn')

trees = pd.read_csv('fff.csv')
trees.shape
trees.head(5)
print(trees.head(5))