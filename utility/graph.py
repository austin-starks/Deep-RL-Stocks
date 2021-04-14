import matplotlib.pyplot as plt
import pandas as pd

def normalize_stock_date(portfolio, stock):
    starting_date = portfolio['Date'][0]
    ending_date = portfolio['Date'][portfolio.index[-1]]
    space_index = starting_date.find(' ')
    space_end_index = starting_date.find(' ')
    if space_index != -1:
        starting_date = starting_date[:space_index]
    if space_end_index != -1:
        ending_date = ending_date[:space_index]
  
    stock_filtered = stock[stock['Date']>= starting_date]
    stock_filtered = stock_filtered[stock_filtered['Date']<= ending_date]

    return stock_filtered


def get_data(filename):
    df = pd.read_csv(filename)
    return df

def combine(portfolio, stock):
    # print(portfolio)
    # print(stock)
    starting_amount = portfolio.iloc[0]['Portfolio Value']
    #2 Factor = 1 + ( Close - Adjusted Close ) / Adjusted Close; Adjusted Open = Open / Factor
    factor = 1 + (stock.iloc[0]['Close'] - stock.iloc[0]['Adj Close']) / stock.iloc[0]['Adj Close']
    adj_open = stock.iloc[0]['Open'] 
    amt_shares = starting_amount / adj_open
    SPY_portfolio = []
    # print(stock)
    for i, row in portfolio.iterrows():
        date = row['Date']  
        space = date.find(' ')
        if space != -1:
            date = date[:space]
        share = stock[stock['Date'] == date]
        if '9:30' in row['Date']:
            close = share['Open'].item()
            # adj_close = share['Adj Close'].item()
            # factor = 1 + (close - adj_close) / adj_close
            # print(factor)
            # adj_open = share['Open'].item() / factor
            amt = amt_shares * close
            SPY_portfolio.append(amt)
        else:
            amt = amt_shares * share['Close'].item()
            SPY_portfolio.append(amt)
    SPY_portfolio = pd.Series(SPY_portfolio, name='SPY Portfolio')
    portfolio = pd.concat([portfolio, SPY_portfolio], axis=1)
    print(portfolio)
    # print(SPY_portfolio.head(20))
    return portfolio

def plot(portfolio):
    ax = portfolio.plot(title='Portfilio Change Over Time', fontsize=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc = 'upper left')
    # plt.axhline(y=0, color='r', linestyle='-')
    plt.show()


if __name__ == "__main__":
    portfolio = get_data("results/test_results.csv")
    stock = get_data("data/price_data/SPY.csv")
    stock = normalize_stock_date(portfolio, stock)

    combined = combine(portfolio, stock)
    plot(combined)



