import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_correlation_calculation():
    """Test correlation calculation with synthetic data to verify our method"""
    
    print("=== Testing Correlation Calculation with Synthetic Data ===")
    
    # Create synthetic data with known correlation
    np.random.seed(42)
    n = 1000
    
    # Create two series with high correlation (0.95)
    x = np.random.normal(0, 1, n)
    noise = np.random.normal(0, 0.1, n)  # Small noise
    y = 0.95 * x + noise
    
    # Calculate correlation
    corr = np.corrcoef(x, y)[0, 1]
    print(f"True correlation: {corr:.4f}")
    
    # Test rolling correlation
    df = pd.DataFrame({'x': x, 'y': y})
    rolling_corr = df['x'].rolling(window=30, min_periods=15).corr(df['y'])
    
    print(f"Rolling correlation (30-day window):")
    print(f"  Mean: {rolling_corr.mean():.4f}")
    print(f"  Max: {rolling_corr.max():.4f}")
    print(f"  Min: {rolling_corr.min():.4f}")
    print(f"  Values > 0.9: {(rolling_corr > 0.9).sum()}")
    print(f"  Values > 0.95: {(rolling_corr > 0.95).sum()}")
    
    # Test with different window sizes
    windows = [15, 30, 60, 90]
    for window in windows:
        corr_window = df['x'].rolling(window=window, min_periods=window//2).corr(df['y'])
        print(f"  {window}-day window - Max: {corr_window.max():.4f}, >0.9: {(corr_window > 0.9).sum()}")
    
    return df, rolling_corr

def fetch_recent_data():
    """Fetch recent data to test correlation calculations"""
    
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("Error: POLYGON_API_KEY environment variable not set")
        return None
    
    # Fetch last 6 months of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    print(f"\n=== Fetching Recent Data ({start_date} to {end_date}) ===")
    
    data = {}
    symbols = ['BTC', 'ETH', 'SPY']
    
    for symbol in symbols:
        base_url = "https://api.polygon.io"
        
        if symbol in ['BTC', 'ETH']:
            url = f"{base_url}/v2/aggs/ticker/X:{symbol}USD/range/1/day/{start_date}/{end_date}"
        else:
            url = f"{base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        
        params = {
            'apiKey': api_key,
            'adjusted': 'true',
            'sort': 'asc'
        }
        
        try:
            print(f"Fetching {symbol}...")
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            result = response.json()
            if result.get('results'):
                df = pd.DataFrame(result['results'])
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df = df[['date', 'c']].rename(columns={'c': 'close'})
                df.set_index('date', inplace=True)
                data[symbol] = df
                print(f"  {symbol}: {len(df)} data points")
            else:
                print(f"  {symbol}: No data")
                
        except Exception as e:
            print(f"  {symbol}: Error - {e}")
    
    return data

def analyze_recent_correlations(data):
    """Analyze correlations in recent data"""
    
    if not data or len(data) < 3:
        print("Insufficient data for correlation analysis")
        return
    
    print("\n=== Analyzing Recent Correlations ===")
    
    # Create combined dataset
    btc_dates = data['BTC'].index
    combined_df = pd.DataFrame(index=btc_dates)
    combined_df['BTC'] = data['BTC']['close']
    combined_df['ETH'] = data['ETH']['close']
    combined_df['SPY'] = data['SPY']['close'].reindex(btc_dates, method='ffill')
    
    combined_df = combined_df.dropna()
    print(f"Combined data: {len(combined_df)} observations")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    
    # Calculate returns
    returns_df = combined_df.pct_change().dropna()
    print(f"Returns data: {len(returns_df)} observations")
    
    # Calculate overall correlations
    btc_spy_corr = returns_df['BTC'].corr(returns_df['SPY'])
    eth_spy_corr = returns_df['ETH'].corr(returns_df['SPY'])
    btc_eth_corr = returns_df['BTC'].corr(returns_df['ETH'])
    
    print(f"\nOverall correlations (entire period):")
    print(f"  BTC vs SPY: {btc_spy_corr:.4f}")
    print(f"  ETH vs SPY: {eth_spy_corr:.4f}")
    print(f"  BTC vs ETH: {btc_eth_corr:.4f}")
    
    # Calculate rolling correlations
    windows = [15, 30, 60, 90]
    
    for window in windows:
        print(f"\n{window}-day rolling correlations:")
        
        # BTC vs SPY
        btc_spy_rolling = returns_df['BTC'].rolling(window=window, min_periods=window//2).corr(returns_df['SPY'])
        valid_btc = btc_spy_rolling.dropna()
        
        if len(valid_btc) > 0:
            print(f"  BTC vs SPY:")
            print(f"    Max: {valid_btc.max():.4f}")
            print(f"    Min: {valid_btc.min():.4f}")
            print(f"    Mean: {valid_btc.mean():.4f}")
            print(f"    > 0.9: {(valid_btc > 0.9).sum()} ({100*(valid_btc > 0.9).sum()/len(valid_btc):.1f}%)")
            print(f"    > 0.8: {(valid_btc > 0.8).sum()} ({100*(valid_btc > 0.8).sum()/len(valid_btc):.1f}%)")
            print(f"    > 0.7: {(valid_btc > 0.7).sum()} ({100*(valid_btc > 0.7).sum()/len(valid_btc):.1f}%)")
            
            # Show recent values
            recent = valid_btc.tail(5)
            print(f"    Recent values: {recent.values}")
        
        # ETH vs SPY
        eth_spy_rolling = returns_df['ETH'].rolling(window=window, min_periods=window//2).corr(returns_df['SPY'])
        valid_eth = eth_spy_rolling.dropna()
        
        if len(valid_eth) > 0:
            print(f"  ETH vs SPY:")
            print(f"    Max: {valid_eth.max():.4f}")
            print(f"    Min: {valid_eth.min():.4f}")
            print(f"    Mean: {valid_eth.mean():.4f}")
            print(f"    > 0.9: {(valid_eth > 0.9).sum()} ({100*(valid_eth > 0.9).sum()/len(valid_eth):.1f}%)")
            print(f"    > 0.8: {(valid_eth > 0.8).sum()} ({100*(valid_eth > 0.8).sum()/len(valid_eth):.1f}%)")
            print(f"    > 0.7: {(valid_eth > 0.7).sum()} ({100*(valid_eth > 0.7).sum()/len(valid_eth):.1f}%)")
    
    return returns_df, btc_spy_rolling, eth_spy_rolling

def plot_recent_correlations(returns_df, btc_corr, eth_corr):
    """Plot recent correlations"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Returns
    ax1.plot(returns_df.index, returns_df['SPY'], label='S&P 500', alpha=0.7, linewidth=1)
    ax1.plot(returns_df.index, returns_df['BTC'], label='Bitcoin', alpha=0.7, linewidth=1)
    ax1.plot(returns_df.index, returns_df['ETH'], label='Ethereum', alpha=0.7, linewidth=1)
    
    ax1.set_title('Daily Returns (Last 6 Months)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rolling correlations
    ax2.plot(btc_corr.index, btc_corr, label='BTC vs S&P 500', linewidth=2, color='blue')
    ax2.plot(eth_corr.index, eth_corr, label='ETH vs S&P 500', linewidth=2, color='orange')
    ax2.axhline(y=0.9, color='red', linestyle=':', alpha=0.7, linewidth=1.5, label='90% Correlation')
    ax2.axhline(y=0.8, color='green', linestyle=':', alpha=0.5, linewidth=1, label='80% Correlation')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
    
    ax2.set_title('30-Day Rolling Correlation')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run correlation tests"""
    
    print("Starting correlation analysis tests...")
    
    # Test 1: Synthetic data
    test_df, test_corr = test_correlation_calculation()
    
    # Test 2: Real data
    data = fetch_recent_data()
    if data:
        returns_df, btc_corr, eth_corr = analyze_recent_correlations(data)
        
        # Plot results
        plot_recent_correlations(returns_df, btc_corr, eth_corr)
    
    print("\nCorrelation tests completed!")

if __name__ == "__main__":
    main()
