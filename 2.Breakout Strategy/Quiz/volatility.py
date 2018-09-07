import pandas as pd
import math
import numpy as np

prices = pd.read_csv("E:/NJIT/AI for trading/2.Breakout Strategy/Quiz/prices.csv")

A = []
A =  prices[prices['ticker'] == 'A'].values
priceA = A[:,0]
price_A = priceA.astype(float)
lretsA = np.log(prices['price']/prices['price'].shift(1))
std_dev_A = lretsA.std()

B = []
B =  prices[prices['ticker'] == 'B'].values
priceB = B[:,0]
price_B = priceB.astype(float)
lretsB = np.log(prices['price']/prices['price'].shift(1))
std_dev_B = lretsB.std()

ticker = 'A' if std_dev_A > std_dev_B else 'B'

print(ticker)