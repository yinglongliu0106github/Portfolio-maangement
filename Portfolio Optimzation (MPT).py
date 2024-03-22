import pandas
import datetime as dt
import numpy as np
import scipy.optimize as sc
from pandas_datareader import data as pdr
import yfinance as yf

def getData(stocks,start,end):
    yf.pdr_override()
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData = stockData['Close']

    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*252
    std = np.sqrt(np.dot(weights.T,np.dot(covMatrix,weights))) * np.sqrt(252)
    return returns,std

stockList = ['CBA','BHP','TLS']
stocks = [stock+'.AX' for stock in stockList]
endDate = dt.datetime.now()
formatted_end_date = endDate.strftime('%Y-%m-%d')
startDate = endDate-dt.timedelta(days = 365)
formatted_start_date = startDate.strftime('%Y-%m-%d')

weights = np.array([0.3,0.3,0.4])

meanReturns,covMatrix = getData(stocks,start=formatted_start_date,end = formatted_end_date)
returns,std = portfolioPerformance(weights,meanReturns,covMatrix)

print(round(returns*100,2),round(std*100,2))

def negativeSR(weights,meanReturns,covMatrix, riskFreeRate = 0):
    preturns,pstd = portfolioPerformance(weights,meanReturns,covMatrix)
    return - (preturns - riskFreeRate)/pstd

def maxSR(meanReturns,covMatrix,riskFreeRate = 0, constraintSet=(0,1)):
    #'Minimize the negative SR, by altering the weights of the portfolio'
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type':'eq','fun':lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(negativeSR, numAssets * [1./numAssets], args = args,
                         method = 'SLSQP', bounds = bounds, constraints = constraints)
    return result

result = maxSR(meanReturns,covMatrix)
maxSR, maxWeights = result['fun'], result['x']
print(maxSR, maxWeights)

# part 3 will be continued