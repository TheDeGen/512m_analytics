"""
Stablecoin Prime Rate (SPR) Plotter Module

This module creates visualizations for DeFi Prime Rate data including
weighted APY trends, pool contributions, and time-series analysis.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
import seaborn as sns

from config import (
    DEFAULT_DB_FILENAME, THEME_PALETTE, MUTED_BLUES, DISPLAY_POOL_NAMES,
    setup_plotting_style
)
from utils import (
    add_logo_overlay, load_data_from_db, format_date_axis,
    validate_dataframe, print_data_summary, create_subplot_grid
)


def plot_weighted_apy_trends(df: pd.DataFrame) -> None:
    """
    Create a line plot showing daily and 14-day moving average weighted APY.
    
    Args:
        df: DataFrame with weighted_apy and ma_apy_14d columns
    """
    if not validate_dataframe(df, ['weighted_apy', 'ma_apy_14d']):
        print("Invalid DataFrame for weighted APY trends plot")
        return
    
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot daily weighted APY
    ax.plot(
        df.index,
        df['weighted_apy'],
        label='Daily Weighted APY',
        alpha=0.6,
        linewidth=1.5,
        color=THEME_PALETTE[2]  # 3rd color in palette
    )
    
    # Plot 14-day moving average
    ax.plot(
        df.index,
        df['ma_apy_14d'],
        label='14-Day Moving Average',
        linewidth=2.5,
        color=THEME_PALETTE[3]  # 4th color in palette
    )
    
    # Customize the plot
    ax.set_title('Stablecoin Prime Rate: Daily vs 14-Day Moving Average')
    ax.set_xlabel('Date')
    ax.set_ylabel('SPR APY (%)')
    ax.legend()
    ax.grid(True)
    
    # Format x-axis dates
    format_date_axis(ax, interval=2)
    
    # Add logo overlay
    add_logo_overlay(ax)
    
    plt.tight_layout()
    plt.show()


def plot_pool_contributions(df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
    """
    Create a figure showing each pool's contribution to the weighted APY.
    
    Args:
        df: DataFrame with pool data
        metadata_df: DataFrame with pool metadata
    """
    if not validate_dataframe(df) or df.empty:
        print("Invalid DataFrame for pool contributions plot")
        return
    
    setup_plotting_style()
    
    # Get the most recent data point
    latest_data = df.iloc[-1]
    
    # Extract APY and TVL columns
    apy_cols = [col for col in df.columns if col.startswith('apy_')]
    tvl_cols = [col for col in df.columns if col.startswith('tvlUsd_')]
    
    # Calculate contributions for the latest data point
    contributions, pool_names = _calculate_pool_contributions(latest_data, apy_cols, tvl_cols, metadata_df)
    
    if not contributions:
        print("No valid contributions to plot")
        return
    
    # Sort by contribution (descending)
    sorted_data = sorted(zip(contributions, pool_names), reverse=True)
    contributions, pool_names = zip(*sorted_data)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Top 15 pools by contribution
    top_n = min(15, len(contributions))
    bar_colors = [MUTED_BLUES[i % len(MUTED_BLUES)] for i in range(top_n)]
    
    bars1 = ax1.bar(range(top_n), contributions[:top_n], color=bar_colors, alpha=0.8)
    ax1.set_title(f'Top {top_n} Pools by Contribution to Weighted APY')
    ax1.set_xlabel('Pool')
    ax1.set_ylabel('Contribution (%)')
    ax1.set_xticks(range(top_n))
    ax1.set_xticklabels([name.replace('Pool_', 'P') for name in pool_names[:top_n]], 
                        rotation=45, ha='right')
    
    # Plot 2: Cumulative contribution
    cumulative = np.cumsum(contributions)
    ax2.plot(
        range(1, len(cumulative) + 1),
        cumulative,
        marker='o',
        linewidth=2,
        markersize=4,
        color=MUTED_BLUES[2]
    )
    ax2.set_title('Cumulative Contribution to Weighted APY')
    ax2.set_xlabel('Number of Pools')
    ax2.set_ylabel('Cumulative Contribution (%)')
    ax2.grid(True)
    
    # Add logo overlay to both subplots
    add_logo_overlay(ax1)
    add_logo_overlay(ax2)
    
    plt.tight_layout()
    plt.show()


def _calculate_pool_contributions(latest_data: pd.Series, apy_cols: List[str], 
                                 tvl_cols: List[str], metadata_df: pd.DataFrame) -> Tuple[List[float], List[str]]:
    """
    Calculate pool contributions to weighted APY.
    
    Args:
        latest_data: Latest data point from merged DataFrame
        apy_cols: List of APY column names
        tvl_cols: List of TVL column names
        metadata_df: DataFrame with pool metadata
        
    Returns:
        Tuple of (contributions list, pool names list)
    """
    contributions = []
    pool_names = []
    
    for i, apy_col in enumerate(apy_cols):
        if i < len(tvl_cols):
            tvl_col = tvl_cols[i]
            apy_val = latest_data[apy_col]
            tvl_val = latest_data[tvl_col]
            
            if pd.notna(apy_val) and pd.notna(tvl_val) and tvl_val > 0:
                # Calculate contribution as (APY * TVL) / total_weighted_sum
                total_weighted_sum = latest_data['weighted_apy'] * latest_data[tvl_cols].sum()
                if total_weighted_sum > 0:
                    contribution = (apy_val * tvl_val) / total_weighted_sum
                    contributions.append(contribution * 100)  # Convert to percentage
                    
                    # Get pool name from metadata
                    pool_num = apy_col.replace('apy_Pool_', '')
                    pool_name = f'Pool_{pool_num}'
                    if len(metadata_df[metadata_df['name'] == pool_name]) > 0:
                        pool_name = metadata_df[metadata_df['name'] == pool_name]['name'].iloc[0]
                    pool_names.append(pool_name)
    
    return contributions, pool_names


def plot_pool_contributions_over_time(df: pd.DataFrame, metadata_df: pd.DataFrame, 
                                     top_n: int = 7) -> None:
    """
    Create a stacked 100% area chart showing the top pools individually and the rest.
    
    Args:
        df: DataFrame with pool data
        metadata_df: DataFrame with pool metadata
        top_n: Number of top pools to show individually
    """
    if not validate_dataframe(df):
        print("Invalid DataFrame for pool contributions over time plot")
        return
    
    setup_plotting_style()
    
    # Extract APY and TVL columns
    apy_cols = [col for col in df.columns if col.startswith('apy_')]
    tvl_cols = [col for col in df.columns if col.startswith('tvlUsd_')]
    
    # Calculate contributions over time for each pool
    pool_contributions = _calculate_time_series_contributions(df, apy_cols, tvl_cols)
    
    # Find top pools by contribution at the latest date
    top_pools = _find_top_contributing_pools(pool_contributions, top_n)
    
    # Create the stacked area chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for stacked area chart
    stack_data = _prepare_stack_data(pool_contributions, top_pools, df)
    
    # Create stacked area plot with consistent colors
    stack_colors = _get_stack_colors(len(stack_data))
    
    ax.stackplot(
        df.index,
        stack_data.values(),
        labels=stack_data.keys(),
        colors=stack_colors,
        alpha=0.9
    )
    
    ax.set_title('Pool Contributions to Stablecoin Prime Rate Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Contribution (%)')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=True, shadow=True)
    ax.grid(True)
    ax.set_ylim(0, 100)
    
    # Format x-axis dates
    format_date_axis(ax, interval=4)
    
    # Add logo overlay
    add_logo_overlay(ax)
    
    plt.tight_layout()
    plt.show()


def _calculate_time_series_contributions(df: pd.DataFrame, apy_cols: List[str], 
                                       tvl_cols: List[str]) -> Dict[str, List[float]]:
    """
    Calculate contributions over time for each pool.
    
    Args:
        df: DataFrame with pool data
        apy_cols: List of APY column names
        tvl_cols: List of TVL column names
        
    Returns:
        Dictionary mapping pool numbers to contribution time series
    """
    pool_contributions = {}
    
    for i, apy_col in enumerate(apy_cols):
        if i < len(tvl_cols):
            tvl_col = tvl_cols[i]
            pool_num = apy_col.replace('apy_Pool_', '')
            
            # Calculate contributions for each time point
            contributions = []
            for idx in range(len(df)):
                apy_val = df.iloc[idx][apy_col]
                tvl_val = df.iloc[idx][tvl_col]
                weighted_apy = df.iloc[idx]['weighted_apy']
                total_tvl = df.iloc[idx][tvl_cols].sum()
                
                if (pd.notna(apy_val) and pd.notna(tvl_val) and tvl_val > 0 and 
                    total_tvl > 0 and weighted_apy > 0):
                    contribution = (apy_val * tvl_val) / (weighted_apy * total_tvl) * 100
                else:
                    contribution = 0
                contributions.append(contribution)
            
            pool_contributions[pool_num] = contributions
    
    return pool_contributions


def _find_top_contributing_pools(pool_contributions: Dict[str, List[float]], 
                               top_n: int) -> List[Tuple[str, float]]:
    """
    Find top pools by contribution at the latest date.
    
    Args:
        pool_contributions: Dictionary of pool contributions over time
        top_n: Number of top pools to return
        
    Returns:
        List of tuples (pool_num, latest_contribution)
    """
    latest_contributions = {}
    latest_idx = len(next(iter(pool_contributions.values()))) - 1
    
    for pool_num, contributions in pool_contributions.items():
        if latest_idx < len(contributions):
            latest_contributions[pool_num] = contributions[latest_idx]
    
    # Sort by latest contribution and get top N
    return sorted(latest_contributions.items(), key=lambda x: x[1], reverse=True)[:top_n]


def _prepare_stack_data(pool_contributions: Dict[str, List[float]], 
                       top_pools: List[Tuple[str, float]], 
                       df: pd.DataFrame) -> Dict[str, List[float]]:
    """
    Prepare data for stacked area chart.
    
    Args:
        pool_contributions: Dictionary of pool contributions over time
        top_pools: List of top contributing pools
        df: Original DataFrame for calculating "Other" category
        
    Returns:
        Dictionary mapping display names to contribution time series
    """
    stack_data = {}
    
    # Add top N pools individually (reverse order so top contributors appear at top of stack)
    for i, (pool_num, latest_contrib) in enumerate(reversed(top_pools)):
        contributions = pool_contributions[pool_num]
        
        # Use custom names for specific pools
        display_name = DISPLAY_POOL_NAMES.get(pool_num, f'Pool_{pool_num}')
        stack_data[f'{display_name} ({latest_contrib:.1f}%)'] = contributions
    
    # Calculate "Other pools" contribution
    other_contributions = np.zeros(len(df))
    for idx in range(len(df)):
        total_contribution = 0
        for pool_num, _ in top_pools:
            contributions = pool_contributions[pool_num]
            if idx < len(contributions):
                total_contribution += contributions[idx]
        other_contributions[idx] = 100 - total_contribution
    
    stack_data['Other Pools'] = other_contributions
    return stack_data


def _get_stack_colors(num_categories: int) -> List[str]:
    """
    Get consistent colors for stacked area chart.
    
    Args:
        num_categories: Number of categories to color
        
    Returns:
        List of color strings
    """
    stack_colors = []
    
    # Use blue hues for all named pools
    for i in range(num_categories - 1):  # All except "Other Pools"
        stack_colors.append(MUTED_BLUES[i % len(MUTED_BLUES)])
    
    # Add specific brown color for "Other Pools"
    stack_colors.append(THEME_PALETTE[2])  # Medium brown
    
    return stack_colors


def create_all_plots(df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
    """
    Create all available plots for the SPR analysis.
    
    Args:
        df: DataFrame with pool data
        metadata_df: DataFrame with pool metadata
    """
    print("\nCreating plots...")
    
    # Plot 1: Weighted APY trends
    print("Creating weighted APY trends plot...")
    plot_weighted_apy_trends(df)
    
    # Plot 2: Pool contributions
    print("Creating pool contributions plot...")
    plot_pool_contributions(df, metadata_df)
    
    # Plot 3: Pool contributions over time
    print("Creating pool contributions over time plot...")
    plot_pool_contributions_over_time(df, metadata_df, top_n=7)


def print_analysis_summary(df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
    """
    Print comprehensive analysis summary statistics.
    
    Args:
        df: DataFrame with pool data
        metadata_df: DataFrame with pool metadata
    """
    print("\n=== Summary Statistics ===")
    
    if 'weighted_apy' in df.columns:
        current_apy = df['weighted_apy'].iloc[-1]
        mean_apy = df['weighted_apy'].mean()
        std_apy = df['weighted_apy'].std()
        min_apy = df['weighted_apy'].min()
        max_apy = df['weighted_apy'].max()
        
        print(f"Current DeFi Prime Rate: {current_apy:.4f}%")
        print(f"Mean DeFi Prime Rate: {mean_apy:.4f}%")
        print(f"Standard Deviation: {std_apy:.4f}%")
        print(f"Range: {min_apy:.4f}% - {max_apy:.4f}%")
    
    print(f"Total pools in dataset: {len(metadata_df)}")
    print("\nPlots created successfully!")


def main() -> None:
    """
    Main function to load data and create plots.
    """
    # Load data from database
    df, metadata_df = load_data_from_db(DEFAULT_DB_FILENAME)
    
    if df is None or metadata_df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Print data summaries
    print_data_summary(df, "Pool Data")
    print_data_summary(metadata_df, "Pool Metadata")
    
    # Create all plots
    create_all_plots(df, metadata_df)
    
    # Print analysis summary
    print_analysis_summary(df, metadata_df)


if __name__ == "__main__":
    main()
