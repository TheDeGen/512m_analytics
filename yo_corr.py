"""
yoUSD Correlation Analysis Module

This module analyzes the correlation between yoUSD APY and the DeFi Prime Rate,
creating comprehensive visualizations including time series, scatter plots,
rolling correlations, and beta analysis.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import time

from config import (
    API_ENDPOINTS, DEFAULT_DB_FILENAME, YOUSD_POOL_ID, YOUSD_POOL_NAME,
    THEME_PALETTE, ROLLING_WINDOW_SIZES, setup_plotting_style
)
from utils import (
    fetch_pool_chart_data, load_data_from_db, add_logo_overlay,
    format_date_axis, validate_dataframe, normalize_datetime_index,
    calculate_rolling_beta, print_data_summary
)


def load_weighted_apy_from_db(db_filename: str = DEFAULT_DB_FILENAME) -> Optional[pd.DataFrame]:
    """
    Load weighted APY data from SQLite database.
    
    Args:
        db_filename: SQLite database filename
        
    Returns:
        DataFrame with weighted APY data, or None if failed
    """
    try:
        print(f"Loading weighted APY data from {db_filename}...")
        conn = sqlite3.connect(db_filename)
        
        # Load main data
        merged_df = pd.read_sql('SELECT date, weighted_apy FROM pool_data', conn)
        
        # Set date as index
        if 'date' in merged_df.columns:
            merged_df['date'] = pd.to_datetime(merged_df['date'])
            merged_df.set_index('date', inplace=True)
        
        conn.close()
        
        print(f"Successfully loaded weighted APY data")
        print_data_summary(merged_df, "Weighted APY Data")
        
        return merged_df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None


def prepare_analysis_data(pool_df: pd.DataFrame, weighted_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Prepare and calculate all analysis data for plotting.
    
    Args:
        pool_df: DataFrame with pool APY data
        weighted_df: DataFrame with weighted APY data
        
    Returns:
        Dictionary containing all calculated data and statistics, or None if failed
    """
    print(f"Preparing analysis data...")
    
    if not validate_dataframe(pool_df) or not validate_dataframe(weighted_df):
        print("Invalid input DataFrames")
        return None
    
    # Normalize datetime indices
    pool_df_normalized = normalize_datetime_index(pool_df)
    weighted_df_normalized = normalize_datetime_index(weighted_df)
    
    # Merge the dataframes on date
    merged_df = pd.merge(pool_df_normalized, weighted_df_normalized, 
                        left_index=True, right_index=True, how='inner')
    
    if merged_df.empty:
        print("No overlapping data found between pool and weighted APY")
        return None
    
    print(f"Found {len(merged_df)} overlapping data points")
    
    # Calculate rolling metrics
    window_size = ROLLING_WINDOW_SIZES['medium']
    merged_df['rolling_corr_30d'] = merged_df['pool_apy'].rolling(
        window=window_size, min_periods=window_size//2
    ).corr(merged_df['weighted_apy'])
    
    merged_df['rolling_beta_30d'] = calculate_rolling_beta(
        merged_df['pool_apy'], merged_df['weighted_apy'], 
        window=window_size, min_periods=window_size//2
    )
    
    # Calculate overall correlation
    correlation = merged_df['pool_apy'].corr(merged_df['weighted_apy'])
    
    # Prepare trend analysis
    trend_data = _calculate_trend_analysis(merged_df)
    
    # Calculate summary statistics
    stats = _calculate_summary_statistics(merged_df, correlation)
    
    return {
        'merged_df': merged_df,
        'trend_data': trend_data,
        'stats': stats
    }


def _calculate_trend_analysis(merged_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate trend line and R-squared for scatter plot analysis.
    
    Args:
        merged_df: Merged DataFrame with pool and weighted APY data
        
    Returns:
        Dictionary containing trend analysis results
    """
    x = merged_df['weighted_apy'].values
    y = merged_df['pool_apy'].values
    
    # Remove NaN values for trend line calculation
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    trend_data = {}
    if len(x_clean) > 1:
        # Calculate linear regression
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        
        # Calculate R-squared
        y_pred = p(x_clean)
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        trend_data = {
            'coefficients': z,
            'polynomial': p,
            'r_squared': r_squared,
            'x_clean': x_clean,
            'y_clean': y_clean
        }
    
    return trend_data


def _calculate_summary_statistics(merged_df: pd.DataFrame, correlation: float) -> Dict[str, Any]:
    """
    Calculate comprehensive summary statistics.
    
    Args:
        merged_df: Merged DataFrame with analysis data
        correlation: Overall correlation coefficient
        
    Returns:
        Dictionary containing summary statistics
    """
    return {
        'data_points': len(merged_df),
        'overall_correlation': correlation,
        'current_pool_apy': merged_df['pool_apy'].iloc[-1],
        'mean_pool_apy': merged_df['pool_apy'].mean(),
        'current_weighted_apy': merged_df['weighted_apy'].iloc[-1],
        'mean_weighted_apy': merged_df['weighted_apy'].mean(),
        'current_rolling_corr': merged_df['rolling_corr_30d'].iloc[-1] if 'rolling_corr_30d' in merged_df.columns else None,
        'mean_rolling_corr': merged_df['rolling_corr_30d'].mean() if 'rolling_corr_30d' in merged_df.columns else None,
        'current_rolling_beta': merged_df['rolling_beta_30d'].iloc[-1] if 'rolling_beta_30d' in merged_df.columns else None,
        'mean_rolling_beta': merged_df['rolling_beta_30d'].mean() if 'rolling_beta_30d' in merged_df.columns else None
    }


def plot_pool_vs_weighted_apy(pool_df: pd.DataFrame, weighted_df: pd.DataFrame, 
                             pool_name: str) -> None:
    """
    Create a comprehensive analysis plot comparing pool APY vs weighted APY.
    
    Args:
        pool_df: DataFrame with pool APY data
        weighted_df: DataFrame with weighted APY data
        pool_name: Name of the pool for display
    """
    # Prepare all analysis data
    analysis_data = prepare_analysis_data(pool_df, weighted_df)
    
    if analysis_data is None:
        print("Failed to prepare analysis data. Exiting.")
        return
    
    merged_df = analysis_data['merged_df']
    trend_data = analysis_data['trend_data']
    stats = analysis_data['stats']
    
    setup_plotting_style()
    
    # Create comprehensive multi-panel plot
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2)
    
    # Panel 1: Time series comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_time_series_comparison(ax1, merged_df)
    
    # Panel 2: Rolling correlation (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    _plot_rolling_correlation(ax2, merged_df)
    
    # Panel 3: Rolling beta (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    _plot_rolling_beta(ax3, merged_df)
    
    # Panel 4: Scatter plot with trend line (top right)
    ax4 = fig.add_subplot(gs[0, 1])
    _plot_scatter_with_trend(ax4, merged_df, trend_data)
    
    # Add logo overlays to all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        add_logo_overlay(ax, alpha=0.04)
    
    plt.tight_layout()
    
    # Save and display plot
    _save_and_display_plot(fig, pool_name, stats, trend_data)


def _plot_time_series_comparison(ax: plt.Axes, merged_df: pd.DataFrame) -> None:
    """Plot time series comparison of pool APY vs weighted APY."""
    ax.plot(
        merged_df.index,
        merged_df['pool_apy'],
        label='yoUSD APY',
        alpha=0.8,
        linewidth=1.5,
        color=THEME_PALETTE[2]
    )
    ax.plot(
        merged_df.index,
        merged_df['weighted_apy'],
        label='SPR APY',
        linewidth=2.5,
        color=THEME_PALETTE[3]
    )
    ax.set_title('yoUSD APY vs Stablecoin Prime Rate Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('APY (%)')
    ax.legend()
    ax.grid(True)
    format_date_axis(ax, interval=2)


def _plot_rolling_correlation(ax: plt.Axes, merged_df: pd.DataFrame) -> None:
    """Plot 30-day rolling correlation."""
    ax.plot(
        merged_df.index,
        merged_df['rolling_corr_30d'],
        linewidth=2,
        color=THEME_PALETTE[4]
    )
    ax.set_title('30-Day Rolling Correlation')
    ax.set_xlabel('Date')
    ax.set_ylabel('Correlation')
    ax.grid(True)
    format_date_axis(ax, interval=2)


def _plot_rolling_beta(ax: plt.Axes, merged_df: pd.DataFrame) -> None:
    """Plot 30-day rolling beta."""
    ax.plot(
        merged_df.index,
        merged_df['rolling_beta_30d'],
        linewidth=2,
        color=THEME_PALETTE[4]
    )
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_title('30-Day Rolling Beta')
    ax.set_xlabel('Date')
    ax.set_ylabel('Beta')
    ax.grid(True)
    format_date_axis(ax, interval=2)


def _plot_scatter_with_trend(ax: plt.Axes, merged_df: pd.DataFrame, 
                           trend_data: Dict[str, Any]) -> None:
    """Plot scatter plot with trend line."""
    # Scatter plot
    ax.scatter(
        merged_df['weighted_apy'],
        merged_df['pool_apy'],
        alpha=0.6,
        s=20,
        color=THEME_PALETTE[2]
    )
    
    # Add trend line if available
    if trend_data:
        z = trend_data['coefficients']
        p = trend_data['polynomial']
        x_clean = trend_data['x_clean']
        
        # Plot trend line
        x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax.plot(x_trend, p(x_trend), 
                color=THEME_PALETTE[3], 
                linewidth=2, 
                linestyle='--',
                label=f'Trend: y = {z[0]:.3f}x + {z[1]:.3f}')
    
    ax.set_xlabel('DeFi Prime Rate (%)')
    ax.set_ylabel('yoUSD APY (%)')
    ax.set_title('Daily Values Scatter Plot')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Set better axis ranges for scatter plot
    _set_scatter_axis_ranges(ax, merged_df)


def _set_scatter_axis_ranges(ax: plt.Axes, merged_df: pd.DataFrame) -> None:
    """Set appropriate axis ranges for scatter plot."""
    x_min, x_max = merged_df['weighted_apy'].min(), merged_df['weighted_apy'].max()
    y_min, y_max = merged_df['pool_apy'].min(), merged_df['pool_apy'].max()
    
    # Add some padding to the ranges
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)


def _save_and_display_plot(fig: plt.Figure, pool_name: str, stats: Dict[str, Any], 
                          trend_data: Dict[str, Any]) -> None:
    """Save plot to file and display, then print summary statistics."""
    # Try to show the plot
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot interactively: {e}")
        print("Plot will be saved as file instead.")
    
    # Save the plot
    plot_filename = f"yousd_vs_defi_prime_rate_analysis.pdf"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor=THEME_PALETTE[0])
    print(f"Comprehensive analysis plot saved as: {plot_filename}")
    
    plt.close(fig)
    
    # Print summary statistics
    _print_analysis_summary(pool_name, stats, trend_data)


def _print_analysis_summary(pool_name: str, stats: Dict[str, Any], 
                          trend_data: Dict[str, Any]) -> None:
    """Print comprehensive analysis summary."""
    print(f"\n=== Summary Statistics ===")
    print(f"Pool: {pool_name}")
    print(f"Data points: {stats['data_points']}")
    print(f"Overall correlation with DeFi Prime Rate: {stats['overall_correlation']:.3f}")
    
    # Rolling statistics
    if stats['current_rolling_corr'] is not None:
        print(f"Current 30-day correlation: {stats['current_rolling_corr']:.3f}")
        print(f"Mean 30-day correlation: {stats['mean_rolling_corr']:.3f}")
    
    if stats['current_rolling_beta'] is not None:
        print(f"Current 30-day beta: {stats['current_rolling_beta']:.3f}")
        print(f"Mean 30-day beta: {stats['mean_rolling_beta']:.3f}")
    
    print(f"yoUSD APY - Current: {stats['current_pool_apy']:.4f}%, Mean: {stats['mean_pool_apy']:.4f}%")
    print(f"DeFi Prime Rate - Current: {stats['current_weighted_apy']:.4f}%, Mean: {stats['mean_weighted_apy']:.4f}%")
    
    if trend_data:
        z = trend_data['coefficients']
        r_squared = trend_data['r_squared']
        print(f"Trend line: y = {z[0]:.3f}x + {z[1]:.3f}")
        print(f"R-squared: {r_squared:.3f}")


def main() -> None:
    """
    Main function to fetch pool data and create comparison plot.
    """
    print(f"=== yoUSD vs DeFi Prime Rate Analysis ===")
    print(f"Pool ID: {YOUSD_POOL_ID}")
    print(f"Pool Name: {YOUSD_POOL_NAME}")
    
    # Fetch pool data from DeFiLlama
    print("\n=== Fetching yoUSD Data from DeFiLlama ===")
    pool_df = fetch_pool_chart_data(YOUSD_POOL_ID, YOUSD_POOL_NAME, days=360)
    
    if not validate_dataframe(pool_df):
        print("Failed to fetch yoUSD data. Exiting.")
        return
    
    # Keep only APY column and rename for clarity
    if 'apy' in pool_df.columns:
        pool_df = pool_df[['apy']].copy()
        pool_df.columns = ['pool_apy']
    
    # Load weighted APY data from database
    print("\n=== Loading DeFi Prime Rate Data from Database ===")
    weighted_df = load_weighted_apy_from_db()
    
    if not validate_dataframe(weighted_df):
        print("Failed to load DeFi Prime Rate data. Exiting.")
        return
    
    # Create the comparison plot
    print("\n=== Creating Comprehensive Analysis ===")
    plot_pool_vs_weighted_apy(pool_df, weighted_df, YOUSD_POOL_NAME)
    
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
