import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def fetch_data(tickers, start_date, end_date):
    """
    Fetch adjusted close prices for given tickers.
    
    Parameters:
    - tickers: list of str, e.g., ['AAPL', 'MSFT']
    - start_date: str, e.g., '2020-01-01'
    - end_date: str, e.g., '2023-01-01'
    
    Returns:
    - pd.DataFrame: Multi-index columns with adjusted close prices.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data.dropna()

def test_cointegration(pair_prices):
    """
    Perform Engle-Granger cointegration test on the pair.
    
    Parameters:
    - pair_prices: pd.DataFrame with two columns (stock1, stock2) of prices.
    
    Returns:
    - tuple: (score, pvalue, beta), where beta is the hedge ratio.
    """
    stock1, stock2 = pair_prices.columns
    y = np.log(pair_prices[stock1])  # Log prices for stationarity
    x = np.log(pair_prices[stock2])
    
    # First step: OLS regression y = alpha + beta * x + epsilon
    model = OLS(y, x).fit()
    beta = model.params[0]  # Hedge ratio
    
    # Second step: Test residuals for stationarity (ADF on residuals)
    residuals = model.resid
    score, pvalue, _ = coint(y, x)  # coint returns ADF stats on residuals
    
    return score, pvalue, beta

def compute_spread(log_prices, beta):
    """
    Compute the spread: log(stock1) - beta * log(stock2)
    
    Parameters:
    - log_prices: pd.DataFrame with log prices.
    - beta: float, hedge ratio.
    
    Returns:
    - pd.Series: Spread values.
    """
    stock1, stock2 = log_prices.columns
    spread = log_prices[stock1] - beta * log_prices[stock2]
    return spread

def generate_signals(spread, window=20, entry_threshold=2, exit_threshold=0.5):
    """
    Generate trading signals based on z-score of spread.
    
    Parameters:
    - spread: pd.Series, spread values.
    - window: int, lookback for mean/std.
    - entry_threshold: float, z-score to enter trade (|z| > entry).
    - exit_threshold: float, z-score to exit (|z| < exit).
    
    Returns:
    - pd.DataFrame: Columns 'zscore', 'signal' (1: long pair, -1: short pair, 0: neutral).
    """
    mean_spread = spread.rolling(window=window).mean()
    std_spread = spread.rolling(window=window).std()
    zscore = (spread - mean_spread) / std_spread
    
    signal = pd.Series(0, index=spread.index)
    position = 0
    
    for i in range(1, len(zscore)):
        if zscore.iloc[i] > entry_threshold and position == 0:
            signal.iloc[i] = 1  # Long stock1, short stock2
            position = 1
        elif zscore.iloc[i] < -entry_threshold and position == 0:
            signal.iloc[i] = -1  # Short stock1, long stock2
            position = -1
        elif abs(zscore.iloc[i]) < exit_threshold and position != 0:
            signal.iloc[i] = 0
            position = 0
        else:
            signal.iloc[i] = position
    
    signals = pd.DataFrame({'zscore': zscore, 'signal': signal}, index=spread.index)
    return signals

def backtest(signals, prices, beta):
    """
    Simple backtest: Compute strategy returns assuming equal weighting.
    
    Parameters:
    - signals: pd.DataFrame from generate_signals.
    - prices: pd.DataFrame of original prices (not log).
    - beta: float, hedge ratio for position sizing.
    
    Returns:
    - dict: Performance metrics (total_return, sharpe, max_drawdown).
    """
    stock1, stock2 = prices.columns
    returns1 = prices[stock1].pct_change()
    returns2 = prices[stock2].pct_change()
    
    # Strategy returns: signal * (ret1 - beta * ret2)
    # Note: For long signal (1): long stock1, short beta*stock2
    # For short signal (-1): short stock1, long beta*stock2 -> - (ret1 - beta * ret2)
    strategy_returns = signals['signal'].shift(1) * (returns1 - beta * returns2)
    
    # Drop NaNs
    strategy_returns = strategy_returns.dropna()
    
    cum_returns = (1 + strategy_returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    
    # Sharpe ratio (assuming risk-free rate=0)
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() != 0 else 0
    
    # Max drawdown
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'cumulative_returns': cum_returns
    }

def plot_results(log_prices, spread, signals, cum_returns, stock1, stock2):
    """
    Plot spread, z-score, signals, and cumulative returns.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Spread and mean
    axes[0].plot(spread.index, spread, label='Spread')
    axes[0].plot(spread.index, spread.rolling(20).mean(), label='Mean', color='r')
    axes[0].set_title(f'{stock1}-{stock2} Spread')
    axes[0].legend()
    
    # Plot 2: Z-score and thresholds
    axes[1].plot(signals.index, signals['zscore'], label='Z-Score')
    axes[1].axhline(2, color='g', linestyle='--', label='Entry Threshold')
    axes[1].axhline(-2, color='g', linestyle='--')
    axes[1].axhline(0.5, color='orange', linestyle='--', label='Exit Threshold')
    axes[1].axhline(-0.5, color='orange', linestyle='--')
    axes[1].set_title('Z-Score with Thresholds')
    axes[1].legend()
    
    # Plot 3: Cumulative returns
    axes[2].plot(cum_returns.index, cum_returns, label='Strategy Returns')
    axes[2].set_title('Cumulative Strategy Returns')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main execution function.
    Customize tickers, dates, and parameters here.
    """
    # Parameters
    tickers = ['AAPL', 'MSFT']  # Example cointegrated pair
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    window = 20
    entry_z = 2.0
    exit_z = 0.5
    
    # Fetch data
    prices = fetch_data(tickers, start_date, end_date)
    log_prices = np.log(prices)
    
    stock1, stock2 = prices.columns
    print(f"Testing cointegration for {stock1} and {stock2}")
    
    # Cointegration test
    score, pvalue, beta = test_cointegration(prices)
    print(f"Cointegration score: {score:.4f}, p-value: {pvalue:.4f}, beta (hedge ratio): {beta:.4f}")
    
    if pvalue > 0.05:
        print("Warning: Pair may not be cointegrated (p > 0.05). Consider another pair.")
    
    # Compute spread
    spread = compute_spread(log_prices, beta)
    
    # Generate signals
    signals = generate_signals(spread, window, entry_z, exit_z)
    
    # Backtest
    results = backtest(signals, prices, beta)
    print(f"Backtest Results:")
    print(f"Total Return: {results['total_return']:.4f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {results['max_drawdown']:.4f}")
    
    # Plot
    plot_results(log_prices, spread, signals, results['cumulative_returns'], stock1, stock2)

if __name__ == "__main__":
    main()