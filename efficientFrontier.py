import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

#verstility
stocks = ('AAPL', 'TSLA', 'NVDA', 'META', 'JPM', 'NFLX', 'HD','DIS')
startDate = '2016-05-02'
endDate = '2023-10-12'
iters = 5000

df = pd.DataFrame()
for stock in stocks:
    #1. take series yf object and extract adjusted close
    #2. dumb it down to a list
    #3. take that list and convert it back into a regular dataframe(axis 1 so add as columns and not rows)
    #4 append the new df to the existing df
    df = pd.concat([df, pd.DataFrame(list(yf.download(stock, start=startDate, end=endDate)['Adj Close']), columns=[stock])], axis=1)
#now we have a df with only adjusted closings
#no need for third party libraries that are going to use yahoo data anyway

#take natural log of quotient of todays price/yesterdays price to get daily returns
#i am using natural log because it is industry standard and is a lot more proffessional than simple percent changes
#drop NA because first row will be NA
returns_daily = np.log(df / df.shift(1)).dropna()
returns_annual = returns_daily.mean() * 252 #average of 252 trading days per year

#covariances are what determine risk/volatility
#they are linear algebra heavy: goal in the future is to do it wihtout pandas
cov_daily = returns_daily.cov()
#cov daily should be symmetric because covariences look at the relationship between two independent variables
cov_annual = cov_daily * 252 #same logic as annual returns

# empty lists to store returns, volatility and info of imiginary portfolios
portfolio_returns = []
portfolio_volatility = []
info_list = []

#now we are going to run a monte carlo simulation
#the monte carlo simulation basically uses volatility and returns to make thousands of simulations of stocks
#each simulation will experiement with the different weights of stocks
#these simulations will later be graphed into the efficient frontier

for single_portfolio in range(iters): #we are running this 5000 times
    #array(length 8) with random decimals 0-1
    weights = np.random.random(len(stocks))
    #make it so that they all add up to 1
    weights /= np.sum(weights)
    #matrix multiplication
    returns = round(np.dot(weights, returns_annual),3)
    #more complex matrix multiplication
    volatility = round(np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights))),3)
    
    #everything being added in for graphing purposes
    portfolio_returns.append(returns)
    portfolio_volatility.append(volatility)
    info = ''
    info+="Weights["
    for i in range(len(stocks)):
        info+=stocks[i]
        info+=": "
        info+= str(float(round(weights[i],2)))
        if i!=(len(stocks)-1):
            info+=", "
    info+="]"
    info_list.append(info)

#gradient purposes
sharpe_ratios = np.array(portfolio_returns) / np.array(portfolio_volatility)

#graphing source(x,y,gradient,hoverdata)
portfolio = {'Returns': portfolio_returns,'Volatility': portfolio_volatility,'Sharpe Ratio': sharpe_ratios,'Info': info_list}

#graph it
color_scale = 'Deep'
fig = px.scatter(portfolio, x='Volatility', y='Returns', title='Efficient Frontier',
                 hover_data='Info', color='Sharpe Ratio', color_continuous_scale=color_scale)
fig.update_layout(width=1200, height=1600)
fig.show()