"""
SPR Testing Module

This module contains test functions for correlation calculations and data analysis,
including synthetic data tests and real market data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import requests
import os
from dotenv import load_dotenv

from config import API_ENDPOINTS, setup_plotting_style
from utils import safe_api_request, validate_dataframe, print_data_summary

# Load environment variables
load_dotenv()


def test_correlation_calculation() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Test correlation calculation with synthetic data to verify methodology.
    
    Returns:
        Tuple of (test DataFrame, rolling correlation series)
    """
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
    
    _print_rolling_correlation_stats(rolling_corr)
    _test_different_window_sizes(df)
    
    return df, rolling_corr


def _print_rolling_correlation_stats(rolling_corr: pd.Series) -> None:
    """Print statistics for rolling correlation series."""
    print(f"Rolling correlation (30-day window):")
    print(f"  Mean: {rolling_corr.mean():.4f}")
    print(f"  Max: {rolling_corr.max():.4f}")
    print(f"  Min: {rolling_corr.min():.4f}")
    print(f"  Values > 0.9: {(rolling_corr > 0.9).sum()}")
    print(f"  Values > 0.95: {(rolling_corr > 0.95).sum()}")


def _test_different_window_sizes(df: pd.DataFrame) -> None:
    """Test correlation with different window sizes."""
    windows = [15, 30, 60, 90]
    for window in windows:
        corr_window = df['x'].rolling(window=window, min_periods=window//2).corr(df['y'])
        max_corr = corr_window.max()
        high_corr_count = (corr_window > 0.9).sum()
        print(f"  {window}-day window - Max: {max_corr:.4f}, >0.9: {high_corr_count}")


def fetch_polygon_data(symbol: str, start_date: str, end_date: str, api_key: str) -> Optional[pd.DataFrame]:
    """
    Fetch data from Polygon.io API for a specific symbol.
    
    Args:
        symbol: Stock/crypto symbol (BTC, ETH, SPY, etc.)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        api_key: Polygon API key
        
    Returns:
        DataFrame with price data, or None if failed
    """
    base_url = API_ENDPOINTS['polygon_base']
    
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
            print(f"  {symbol}: {len(df)} data points")
            return df
        else:
            print(f"  {symbol}: No data")
            return None
            
    except Exception as e:
        print(f"  {symbol}: Error - {e}")
        return None


def fetch_recent_data() -> Optional[Dict[str, pd.DataFrame]]:
    """
    Fetch recent market data to test correlation calculations.
    
    Returns:
        Dictionary mapping symbols to DataFrames, or None if failed
    """
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
        df = fetch_polygon_data(symbol, start_date, end_date, api_key)
        if df is not None:
            data[symbol] = df
    
    return data if data else None


def analyze_recent_correlations(data: Dict[str, pd.DataFrame]) -> Optional[Tuple[pd.DataFrame, pd.Series, pd.Series]]:
    """
    Analyze correlations in recent market data.
    
    Args:
        data: Dictionary mapping symbols to DataFrames
        
    Returns:
        Tuple of (returns DataFrame, BTC-SPY correlation, ETH-SPY correlation) or None if failed
    """
    if not data or len(data) < 3:
        print("Insufficient data for correlation analysis")
        return None
    
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
    _print_overall_correlations(returns_df)
    
    # Calculate rolling correlations
    btc_spy_rolling, eth_spy_rolling = _calculate_rolling_correlations(returns_df)
    
    return returns_df, btc_spy_rolling, eth_spy_rolling


def _print_overall_correlations(returns_df: pd.DataFrame) -> None:
    """Print overall correlation statistics."""
    btc_spy_corr = returns_df['BTC'].corr(returns_df['SPY'])
    eth_spy_corr = returns_df['ETH'].corr(returns_df['SPY'])
    btc_eth_corr = returns_df['BTC'].corr(returns_df['ETH'])
    
    print(f"\nOverall correlations (entire period):")
    print(f"  BTC vs SPY: {btc_spy_corr:.4f}")
    print(f"  ETH vs SPY: {eth_spy_corr:.4f}")
    print(f"  BTC vs ETH: {btc_eth_corr:.4f}")


def _calculate_rolling_correlations(returns_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Calculate and print rolling correlations for different window sizes."""
    windows = [15, 30, 60, 90]
    
    # Calculate 30-day rolling correlations for plotting
    btc_spy_rolling = returns_df['BTC'].rolling(window=30, min_periods=15).corr(returns_df['SPY'])
    eth_spy_rolling = returns_df['ETH'].rolling(window=30, min_periods=15).corr(returns_df['SPY'])
    
    for window in windows:
        print(f"\n{window}-day rolling correlations:")
        
        # BTC vs SPY
        btc_spy_window = returns_df['BTC'].rolling(window=window, min_periods=window//2).corr(returns_df['SPY'])
        _print_window_correlation_stats(btc_spy_window, "BTC vs SPY")
        
        # ETH vs SPY
        eth_spy_window = returns_df['ETH'].rolling(window=window, min_periods=window//2).corr(returns_df['SPY'])
        _print_window_correlation_stats(eth_spy_window, "ETH vs SPY")
    
    return btc_spy_rolling, eth_spy_rolling


def _print_window_correlation_stats(correlation_series: pd.Series, label: str) -> None:
    """Print statistics for a windowed correlation series."""
    valid_corr = correlation_series.dropna()
    
    if len(valid_corr) > 0:
        print(f"  {label}:")
        print(f"    Max: {valid_corr.max():.4f}")
        print(f"    Min: {valid_corr.min():.4f}")
        print(f"    Mean: {valid_corr.mean():.4f}")
        print(f"    > 0.9: {(valid_corr > 0.9).sum()} ({100*(valid_corr > 0.9).sum()/len(valid_corr):.1f}%)")
        print(f"    > 0.8: {(valid_corr > 0.8).sum()} ({100*(valid_corr > 0.8).sum()/len(valid_corr):.1f}%)")
        print(f"    > 0.7: {(valid_corr > 0.7).sum()} ({100*(valid_corr > 0.7).sum()/len(valid_corr):.1f}%)")
        
        # Show recent values
        recent = valid_corr.tail(5)
        print(f"    Recent values: {recent.values}")


def plot_recent_correlations(returns_df: pd.DataFrame, btc_corr: pd.Series, 
                           eth_corr: pd.Series) -> None:
    """
    Plot recent correlations and returns data.
    
    Args:
        returns_df: DataFrame with returns data
        btc_corr: BTC-SPY correlation series
        eth_corr: ETH-SPY correlation series
    """
    setup_plotting_style()
    
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


def run_synthetic_data_test() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Run synthetic data correlation test.
    
    Returns:
        Tuple of test DataFrame and correlation series
    """
    print("Running synthetic data correlation test...")
    return test_correlation_calculation()


def run_real_data_test() -> Optional[Tuple[pd.DataFrame, pd.Series, pd.Series]]:
    """
    Run real market data correlation test.
    
    Returns:
        Tuple of (returns DataFrame, BTC correlation, ETH correlation) or None if failed
    """
    print("Running real market data correlation test...")
    
    data = fetch_recent_data()
    if not data:
        print("Failed to fetch market data")
        return None
    
    result = analyze_recent_correlations(data)
    if result:
        returns_df, btc_corr, eth_corr = result
        plot_recent_correlations(returns_df, btc_corr, eth_corr)
        return result
    
    return None


def main() -> None:
    """
    Main function to run all correlation tests.
    """
    print("Starting correlation analysis tests...")
    
    # Test 1: Synthetic data
    print("\n" + "="*50)
    test_df, test_corr = run_synthetic_data_test()
    
    # Test 2: Real data
    print("\n" + "="*50)
    real_data_result = run_real_data_test()
    
    print("\n" + "="*50)
    print("Correlation tests completed!")
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Synthetic data test: {'✓ Passed' if test_df is not None else '✗ Failed'}")
    print(f"Real data test: {'✓ Passed' if real_data_result is not None else '✗ Failed'}")


if __name__ == "__main__":
    main()
