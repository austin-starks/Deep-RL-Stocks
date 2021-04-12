
import matplotlib as plt
import pandas as pd

def get_data(filename):
    df = pd.read_csv(filename, index_col='Date')
    return df

    
def combine(portfolio, stock):
    return portfolio


def plot(df):
    df.plot()
    plt.show()
if __name__ == "__main__":
    portfolio_value = get_data("test_results.csv")
    stock = get_data("SPY.csv")
    df = combine(portfolio_value, stock)
    plot(df)
