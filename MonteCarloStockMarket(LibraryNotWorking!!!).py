# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 20:45:46 2025

@author: lexma
"""

#Test using Monte Carlo method to simualte a stock portfolio

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from pandas_datareader import data as pdr

def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks,start,end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stocklist = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stocklist]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

print(meanReturns)