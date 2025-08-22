"""
Specific Pools Fetcher Module

This module fetches and analyzes data for specific DeFi pools,
creating visualizations comparing pool APY trends with Ethereum price data.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import time

from config import (
    SPECIFIC_POOL_IDS, POOL_NAMES, THEME_PALETTE, MUTED_BLUES,
    DEFAULT_FETCH_DAYS, RATE_LIMIT_DELAY, setup_plotting_style
)
from utils import (
    add_logo_overlay, fetch_pool_chart_data, fetch_ethereum_price_data,
    format_date_axis, validate_dataframe, print_data_summary
)


def plot_pool_apy_trends(pool_data: Dict[str, Dict[str, Any]], 
                        price_data: Optional[pd.DataFrame]) -> None:
    """
    Create a figure with two subplots:
    1) 7-day moving average APY values
    2) Ethereum price
    
    Args:
        pool_data: Dictionary containing pool data with structure {pool_id: {'data': df, 'id': str}}
        price_data: DataFrame containing Ethereum price data
    """
    if not pool_data:
        print("No pool data provided for plotting")
        return
    
    # Set up plotting style
    setup_plotting_style()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: 7-day moving average APY values for each pool
    for i, (pool_id, pool_info) in enumerate(pool_data.items()):
        df = pool_info['data']
        pool_name = POOL_NAMES.get(pool_id, f"Pool_{pool_id}")
        
        # Calculate 7-day moving average of APY
        if validate_dataframe(df, ['apy']):
            ma_apy = df['apy'].rolling(window=7, min_periods=1).mean()
            
            # Use different colors from the muted_blues palette
            color = MUTED_BLUES[i % len(MUTED_BLUES)]
            
            ax1.plot(
                df.index,
                ma_apy,
                label=pool_name,
                linewidth=2,
                alpha=0.8,
                color=color
            )
    
    ax1.set_title('7-Day Moving Average APY on AAVE V3')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('APY (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis dates
    format_date_axis(ax1, interval=3)
    
    # Add logo overlay to first subplot
    add_logo_overlay(ax1)
    
    # Plot 2: Ethereum price
    if validate_dataframe(price_data, ['close']):
        ax2.plot(
            price_data.index,
            price_data['close'],
            label='ETH Price',
            linewidth=2,
            color=THEME_PALETTE[3],  # 4th color in palette
            alpha=0.8
        )
        
        ax2.set_title('Ethereum Price (USD)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price (USD)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        format_date_axis(ax2, interval=3)
        
        # Add logo overlay to second subplot
        add_logo_overlay(ax2)
    else:
        ax2.text(0.5, 0.5, 'Ethereum price data not available', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, color=THEME_PALETTE[4])
        ax2.set_title('Ethereum Price')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()


def fetch_specific_pools_data(days: int = DEFAULT_FETCH_DAYS) -> Dict[str, Dict[str, Any]]:
    """
    Fetch data for all specific pools defined in configuration.
    
    Args:
        days: Number of days of historical data to fetch
        
    Returns:
        Dictionary containing fetched pool data
    """
    print(f"Fetching data for {len(SPECIFIC_POOL_IDS)} specific pools...")
    
    pool_data = {}
    
    for pool_id in SPECIFIC_POOL_IDS:
        pool_name = POOL_NAMES.get(pool_id, pool_id)
        df = fetch_pool_chart_data(pool_id, pool_name, days)
        
        if validate_dataframe(df):
            pool_data[pool_id] = {
                'data': df,
                'id': pool_id
            }
            print_data_summary(df, f"{pool_name} Pool Data")
        
        # Add small delay to avoid rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    return pool_data


def main() -> None:
    """
    Main function to fetch specific pool data and create plots.
    """
    print("=== Specific Pool APY Analysis ===")
    
    # Fetch data for each pool
    pool_data = fetch_specific_pools_data()
    
    if not pool_data:
        print("No pool data fetched successfully. Exiting.")
        return
    
    print(f"\nSuccessfully fetched data for {len(pool_data)} pools")
    
    # Fetch Ethereum price data for comparison
    print("\nFetching Ethereum price data...")
    start_date = datetime.now() - timedelta(days=DEFAULT_FETCH_DAYS)
    end_date = datetime.now()
    price_data = fetch_ethereum_price_data(start_date, end_date)
    
    if price_data is not None:
        print_data_summary(price_data, "Ethereum Price Data")
    
    # Create plots
    print("\nCreating plots...")
    plot_pool_apy_trends(pool_data, price_data)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
