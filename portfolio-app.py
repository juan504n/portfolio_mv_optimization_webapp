import streamlit as st
import pandas as pd
import pandas_datareader.data as web
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import datetime as dt

st.header('''
Portfolio Mean Variance Optimization 
''')

st.sidebar.markdown(''' Change the parameters of your portfolio.''')
st.sidebar.markdown(''' Enter the correct ticker or you will get an error.''')

# Collect user input into dataframe 

def user_input_features():
    start = st.sidebar.date_input('Start date', dt.date(2000,1,1))
    end = st.sidebar.date_input('End date', dt.date(2020,8,8))
    # start = dt.datetime(2000,4,1)
    # end = dt.datetime(2020,4,1)
    stock1 = st.sidebar.text_input("Stock 1 Ticker", 'AMZN')
    stock2 = st.sidebar.text_input("Stock 2 Ticker", 'BRK-B')
    stock3 = st.sidebar.text_input("Stock 3 Ticker", 'JPM')
    stock4 = st.sidebar.text_input("Stock 4 Ticker", 'VFINX')
    stock5 = st.sidebar.text_input("Stock 5 Ticker", 'MCD')
    df1 = web.DataReader(stock1, 'yahoo', start, end)['Adj Close']
    df2 = web.DataReader(stock2, 'yahoo', start, end)['Adj Close']
    df3 = web.DataReader(stock3, 'yahoo', start, end)['Adj Close']
    df4 = web.DataReader(stock4, 'yahoo', start, end)['Adj Close']
    df5 = web.DataReader(stock5, 'yahoo', start, end)['Adj Close']
    portfolio = pd.concat([df1, df2, df3,df4,df5],axis=1)
    portfolio.columns=[stock1, stock2, stock3, stock4, stock5]
    return portfolio
input_df = user_input_features()

# Displays the user input features 
st.subheader('User Input features')


st.write('Adjusted closed price is shown.')
st.write(input_df)


st.subheader('''
Volatility of each stock
''')
returns = input_df.pct_change()
plt.figure(figsize=(15, 8))
for c in returns.columns.values:
    plt.plot(returns.index, returns[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper right', fontsize=12)
plt.ylabel('daily returns')
st.pyplot()

st.subheader('''
Historic Price Action
''')
plt.figure(figsize=(14, 7))
for c in input_df.columns.values:
    plt.plot(input_df.index, input_df[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('Adjusted Closed Price')
st.pyplot()



def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def generate_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    '''Generate portfolios with random weights'''
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(5)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

# parameters
returns = input_df.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 50000
risk_free_rate = 0.0115

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

def display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, _ = generate_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=input_df.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=input_df.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    st.write( "-"*80)
    st.write( "Maximum Sharpe Ratio Portfolio Allocation\n")
    st.write( "Annualised Return:", round(rp,2))
    st.write( "Annualised Volatility:", round(sdp,2))
    st.write( "\n")
    st.write( max_sharpe_allocation)
    st.write( "-"*80)
    st.write( "Minimum Volatility Portfolio Allocation\n")
    st.write( "Annualised Return:", round(rp_min,2))
    st.write( "Annualised Volatility:", round(sdp_min,2))
    st.write( "\n")
    st.write( min_vol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, 0.27, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
    st.pyplot()

display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)