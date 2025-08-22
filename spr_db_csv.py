"""
SPR Database CSV Export Module

This module loads data from the SQLite database, extracts specific pool APY data,
and exports it to CSV format for external analysis.
"""

import pandas as pd
from datetime import datetime
from typing import Optional

from config import DEFAULT_DB_FILENAME
from utils import load_data_from_db, validate_dataframe, print_data_summary


def extract_pool_apy_data(df: pd.DataFrame, pool_number: int = 0) -> Optional[pd.DataFrame]:
    """
    Extract specific pool APY and weighted APY data.
    
    Args:
        df: Full dataset from database
        pool_number: Pool number to extract (default: 0)
        
    Returns:
        DataFrame with pool APY and weighted APY, or None if failed
    """
    # Define column names
    pool_apy_col = f'apy_Pool_{pool_number}'
    weighted_apy_col = 'weighted_apy'
    
    # Check if columns exist
    if pool_apy_col not in df.columns:
        print(f"Warning: {pool_apy_col} not found in dataset")
        available_apy_cols = [col for col in df.columns if col.startswith('apy_')]
        print(f"Available APY columns: {available_apy_cols}")
        return None
    
    if weighted_apy_col not in df.columns:
        print(f"Warning: {weighted_apy_col} not found in dataset")
        return None
    
    # Create subset with pool APY and weighted APY
    result_df = df[[pool_apy_col, weighted_apy_col]].copy()
    result_df.columns = [f'Pool_{pool_number}_APY', 'Weighted_APY']
    
    return result_df


def export_to_csv(pool_data: pd.DataFrame, filename: str = None) -> str:
    """
    Export pool data to CSV file.
    
    Args:
        pool_data: DataFrame to export
        filename: Output filename (optional, will generate if not provided)
        
    Returns:
        Filename of exported CSV
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pool_apy_data_{timestamp}.csv"
    
    pool_data.to_csv(filename)
    print(f"\nData saved to {filename}")
    print(f"CSV file contains {len(pool_data)} rows of data")
    
    return filename


def print_data_analysis(pool_data: pd.DataFrame) -> None:
    """
    Print comprehensive data analysis including head, tail, and statistics.
    
    Args:
        pool_data: DataFrame to analyze
    """
    print("\n=== HEAD (First 10 rows) ===")
    print(pool_data.head(10))
    
    print("\n=== TAIL (Last 10 rows) ===")
    print(pool_data.tail(10))
    
    print("\n=== SUMMARY STATISTICS ===")
    print(pool_data.describe())
    
    print("\n=== CURRENT VALUES ===")
    if not pool_data.empty:
        latest_data = pool_data.iloc[-1]
        for col in pool_data.columns:
            print(f"Latest {col}: {latest_data[col]:.4f}%")


def main() -> None:
    """
    Main function to load data, extract pool information, and export to CSV.
    """
    print("=== SPR Database CSV Export ===")
    
    # Load data from database
    df, metadata_df = load_data_from_db(DEFAULT_DB_FILENAME)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    print_data_summary(df, "Database Data")
    
    # Extract pool 0 APY and weighted APY data
    pool_data = extract_pool_apy_data(df, pool_number=0)
    
    if pool_data is None:
        print("Failed to extract pool data. Exiting.")
        return
    
    # Print analysis
    print_data_analysis(pool_data)
    
    # Export to CSV
    csv_filename = export_to_csv(pool_data, "pool_0_apy_data.csv")
    
    print(f"\n=== Export Complete ===")
    print(f"Data exported to: {csv_filename}")


if __name__ == "__main__":
    main()
