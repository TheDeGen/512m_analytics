"""
Utility functions for DeFi Prime Rate analysis project.

This module contains common functionality used across multiple modules
to eliminate code duplication and improve maintainability.
"""

import sqlite3
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import time
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from config import (
    API_ENDPOINTS, DEFAULT_DB_FILENAME, DEFAULT_LOGO_PATH, DEFAULT_LOGO_ALPHA,
    RATE_LIMIT_DELAY, RATE_LIMIT_RETRY_DELAY, THEME_PALETTE
)


def add_logo_overlay(ax: plt.Axes, logo_path: str = DEFAULT_LOGO_PATH, 
                    alpha: float = DEFAULT_LOGO_ALPHA) -> None:
    """
    Add logo overlay to the center of a matplotlib plot.
    
    Args:
        ax: matplotlib axis object
        logo_path: path to logo image file
        alpha: transparency level (0-1)
    """
    try:
        # Load the logo image
        logo_img = Image.open(logo_path)
        logo_array = np.array(logo_img)
        
        # Get the center of the plot
        x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
        y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
        
        # Calculate appropriate size for the logo (about 30% of plot width)
        plot_width = ax.get_xlim()[1] - ax.get_xlim()[0]
        logo_width = plot_width * 0.3
        
        # Create offset image
        im = OffsetImage(logo_array, zoom=logo_width/logo_img.width, alpha=alpha)
        
        # Create annotation box at center
        ab = AnnotationBbox(im, (x_center, y_center), frameon=False)
        ax.add_artist(ab)
        
    except Exception as e:
        print(f"Warning: Could not add logo overlay: {e}")


def fetch_pool_chart_data(pool_id: str, pool_name: str = None, 
                         days: int = 360) -> Optional[pd.DataFrame]:
    """
    Fetch historical chart data for a specific pool from DeFiLlama API.
    
    Args:
        pool_id: Pool ID from DeFiLlama
        pool_name: Pool name for logging (optional)
        days: Number of days of historical data to fetch
        
    Returns:
        DataFrame with historical APY and TVL data, or None if failed
    """
    display_name = pool_name or pool_id
    
    try:
        print(f"Fetching data for {display_name}...")
        url = f"{API_ENDPOINTS['defi_llama_chart']}{pool_id}"
        response = requests.get(url)
        
        # Handle rate limiting
        if response.status_code == 429:
            print(f"Rate limited for {display_name}, waiting {RATE_LIMIT_RETRY_DELAY} seconds...")
            time.sleep(RATE_LIMIT_RETRY_DELAY)
            response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract data array
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df = _process_timestamp_column(df, display_name)
                    if df is None:
                        return None
                    
                    # Filter to last N days
                    cutoff_date = datetime.now() - timedelta(days=days)
                    df = df[df.index >= cutoff_date]
                    
                    print(f"Successfully fetched {len(df)} data points for {display_name}")
                    return df
                else:
                    print(f"No timestamp found in data for {display_name}")
                    return None
            else:
                print(f"Unexpected data format for {display_name}")
                return None
        else:
            print(f"Error fetching chart data for {display_name}: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching chart data for {display_name}: {e}")
        return None


def _process_timestamp_column(df: pd.DataFrame, pool_name: str) -> Optional[pd.DataFrame]:
    """
    Process timestamp column and set as index.
    
    Args:
        df: DataFrame with timestamp column
        pool_name: Pool name for error logging
        
    Returns:
        Processed DataFrame or None if failed
    """
    try:
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    except (ValueError, TypeError):
        try:
            df['date'] = pd.to_datetime(df['timestamp'])
        except (ValueError, TypeError):
            print(f"Could not parse timestamp format for pool {pool_name}")
            return None
    
    df.set_index('date', inplace=True)
    
    # Make index timezone-naive
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    return df


def fetch_ethereum_price_data(start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """
    Fetch Ethereum price data from Polygon API.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with Ethereum price data, or None if failed
    """
    try:
        api_key = os.getenv('POLYGON_API_KEY')
        
        if not api_key:
            print("Warning: POLYGON_API_KEY not found in environment variables")
            return None
        
        # Convert dates to string format for Polygon API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Polygon API endpoint for Ethereum daily prices
        url = f"{API_ENDPOINTS['polygon_base']}/v2/aggs/ticker/X:ETHUSD/range/1/day/{start_str}/{end_str}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
        
        print(f"Fetching Ethereum price data from {start_str} to {end_str}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('results'):
                # Extract price data
                price_data = []
                for result in data['results']:
                    price_data.append({
                        'date': datetime.fromtimestamp(result['t'] / 1000),
                        'open': result['o'],
                        'high': result['h'],
                        'low': result['l'],
                        'close': result['c'],
                        'volume': result['v']
                    })
                
                df = pd.DataFrame(price_data)
                df.set_index('date', inplace=True)
                
                print(f"Successfully fetched {len(df)} days of Ethereum price data")
                return df
            else:
                print("No results found in Polygon API response")
                return None
        else:
            print(f"Error fetching Ethereum price data: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching Ethereum price data: {e}")
        return None


def load_data_from_db(db_filename: str = DEFAULT_DB_FILENAME) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load data from SQLite database.
    
    Args:
        db_filename: SQLite database filename
        
    Returns:
        Tuple of (merged_df, metadata_df) or (None, None) if failed
    """
    try:
        print(f"Loading data from {db_filename}...")
        conn = sqlite3.connect(db_filename)
        
        # Check what tables exist
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in database: {[table[0] for table in tables]}")
        
        # Load main data
        try:
            merged_df = pd.read_sql('SELECT * FROM pool_data', conn, index_col='index')
        except:
            try:
                merged_df = pd.read_sql('SELECT * FROM pool_data', conn)
                if len(merged_df.columns) > 0:
                    first_col = merged_df.columns[0]
                    if 'date' in first_col.lower() or 'time' in first_col.lower():
                        merged_df.set_index(first_col, inplace=True)
                    else:
                        merged_df.set_index(first_col, inplace=True)
            except Exception as e:
                print(f"Error reading pool_data table: {e}")
                cursor.execute("PRAGMA table_info(pool_data);")
                columns = cursor.fetchall()
                print("Pool_data table structure:")
                for col in columns:
                    print(f"  {col[1]} ({col[2]})")
                return None, None
        
        # Convert index to datetime if needed
        if not isinstance(merged_df.index, pd.DatetimeIndex):
            try:
                merged_df.index = pd.to_datetime(merged_df.index)
            except:
                print("Warning: Could not convert index to datetime")
        
        # Load metadata
        try:
            metadata_df = pd.read_sql('SELECT * FROM pool_metadata', conn)
        except Exception as e:
            print(f"Error reading pool_metadata table: {e}")
            metadata_df = pd.DataFrame()
        
        conn.close()
        
        print(f"Successfully loaded data for {len(metadata_df)} pools")
        return merged_df, metadata_df
        
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None, None


def format_date_axis(ax: plt.Axes, interval: int = 2) -> None:
    """
    Format x-axis dates consistently across plots.
    
    Args:
        ax: matplotlib axis object
        interval: interval for date ticks (in weeks)
    """
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def calculate_rolling_correlation(series1: pd.Series, series2: pd.Series, 
                                window: int = 30, min_periods: int = 15) -> pd.Series:
    """
    Calculate rolling correlation between two pandas Series.
    
    Args:
        series1: First time series
        series2: Second time series  
        window: Rolling window size
        min_periods: Minimum periods required for calculation
        
    Returns:
        Series with rolling correlation values
    """
    return series1.rolling(window=window, min_periods=min_periods).corr(series2)


def calculate_rolling_beta(dependent_series: pd.Series, independent_series: pd.Series,
                          window: int = 30, min_periods: int = 15) -> pd.Series:
    """
    Calculate rolling beta between two series using returns.
    
    Args:
        dependent_series: Dependent variable series
        independent_series: Independent variable series
        window: Rolling window size
        min_periods: Minimum periods required for calculation
        
    Returns:
        Series with rolling beta values
    """
    beta_series = pd.Series(index=dependent_series.index, dtype=float)
    
    for i in range(min_periods-1, len(dependent_series)):
        start_idx = max(0, i - window + 1)
        dep_window = dependent_series.iloc[start_idx:i+1]
        indep_window = independent_series.iloc[start_idx:i+1]
        
        if len(dep_window) >= min_periods:
            # Calculate returns (first difference)
            dep_returns = dep_window.diff().dropna()
            indep_returns = indep_window.diff().dropna()
            
            if len(dep_returns) > 1 and len(indep_returns) > 1:
                # Align the series
                common_idx = dep_returns.index.intersection(indep_returns.index)
                if len(common_idx) > 1:
                    dep_aligned = dep_returns.loc[common_idx]
                    indep_aligned = indep_returns.loc[common_idx]
                    
                    # Calculate beta
                    covariance = np.cov(dep_aligned, indep_aligned)[0, 1]
                    market_variance = np.var(indep_aligned)
                    
                    if market_variance > 0:
                        beta_series.iloc[i] = covariance / market_variance
    
    return beta_series


def purge_database(db_filename: str = DEFAULT_DB_FILENAME) -> None:
    """
    Completely purge the database before fresh data insertion.
    
    Args:
        db_filename: SQLite database filename
    """
    print(f"Purging existing database: {db_filename}")
    try:
        conn = sqlite3.connect(db_filename)
        cursor = conn.cursor()
        
        # Drop existing tables if they exist
        cursor.execute("DROP TABLE IF EXISTS pool_data")
        cursor.execute("DROP TABLE IF EXISTS pool_metadata")
        
        # Clean up any other artifacts
        cursor.execute("VACUUM")
        
        conn.commit()
        conn.close()
        print("Database purged successfully")
    except Exception as e:
        print(f"Warning: Could not purge database: {e}")


def safe_api_request(url: str, max_retries: int = 3) -> Optional[requests.Response]:
    """
    Make API request with rate limiting and retry logic.
    
    Args:
        url: API endpoint URL
        max_retries: Maximum number of retry attempts
        
    Returns:
        Response object or None if failed
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            
            if response.status_code == 429:  # Rate limited
                wait_time = RATE_LIMIT_RETRY_DELAY * (attempt + 1)
                print(f"Rate limited, waiting {wait_time} seconds... (attempt {attempt + 1})")
                time.sleep(wait_time)
                continue
            
            return response
            
        except Exception as e:
            print(f"Request failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(RATE_LIMIT_DELAY)
    
    return None


def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    return True


def normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize datetime index to remove time component and ensure consistency.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with normalized datetime index
    """
    df_copy = df.copy()
    
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy.index = pd.to_datetime(df_copy.index)
    
    df_copy.index = df_copy.index.normalize()
    
    # Make index timezone-naive
    if df_copy.index.tz is not None:
        df_copy.index = df_copy.index.tz_localize(None)
    
    return df_copy


def print_data_summary(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print a standardized summary of DataFrame contents.
    
    Args:
        df: DataFrame to summarize
        name: Name for the dataset in output
    """
    print(f"\n=== {name} Summary ===")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for missing values
    missing_data = df.isnull().sum()
    if missing_data.any():
        print("Missing values:")
        for col, count in missing_data.items():
            if count > 0:
                print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")


def create_subplot_grid(nrows: int, ncols: int, figsize: Tuple[int, int] = (16, 12)) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a standardized subplot grid with consistent styling.
    
    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns  
        figsize: Figure size tuple
        
    Returns:
        Tuple of (figure, axes_array)
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Ensure axes is always an array for consistent handling
    if nrows * ncols == 1:
        axes = np.array([axes])
    elif nrows == 1 or ncols == 1:
        axes = np.array(axes)
    
    return fig, axes


# Export commonly used functions
__all__ = [
    'add_logo_overlay',
    'fetch_pool_chart_data', 
    'fetch_ethereum_price_data',
    'load_data_from_db',
    'format_date_axis',
    'calculate_rolling_correlation',
    'calculate_rolling_beta',
    'purge_database',
    'safe_api_request',
    'validate_dataframe',
    'normalize_datetime_index',
    'print_data_summary',
    'create_subplot_grid'
]