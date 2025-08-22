import sqlite3
import pandas as pd
from datetime import datetime

def load_pool_data(db_filename="defi_prime_rate.db"):
    """
    Load data from SQLite database and extract pool 0 APY and weighted APY
    
    Args:
        db_filename (str): SQLite database filename
        
    Returns:
        pd.DataFrame: DataFrame with pool 0 APY and weighted APY
    """
    try:
        print(f"Loading data from {db_filename}...")
        conn = sqlite3.connect(db_filename)
        
        # Load main data
        df = pd.read_sql('SELECT * FROM pool_data', conn)
        
        # Set date as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        conn.close()
        
        print(f"Successfully loaded data")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Total data points: {len(df)}")
        
        return df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None

def extract_pool_apy_data(df):
    """
    Extract pool 0 APY and weighted APY data
    
    Args:
        df (pd.DataFrame): Full dataset from database
        
    Returns:
        pd.DataFrame: DataFrame with pool 0 APY and weighted APY
    """
    # Extract pool 0 APY and weighted APY columns
    pool_0_apy_col = 'apy_Pool_0'
    weighted_apy_col = 'weighted_apy'
    
    # Check if columns exist
    if pool_0_apy_col not in df.columns:
        print(f"Warning: {pool_0_apy_col} not found in dataset")
        print(f"Available APY columns: {[col for col in df.columns if col.startswith('apy_')]}")
        return None
    
    if weighted_apy_col not in df.columns:
        print(f"Warning: {weighted_apy_col} not found in dataset")
        return None
    
    # Create subset with pool 0 APY and weighted APY
    result_df = df[[pool_0_apy_col, weighted_apy_col]].copy()
    result_df.columns = ['Pool_0_APY', 'Weighted_APY']
    
    return result_df

def main():
    """
    Main function to load data and display pool 0 APY and weighted APY
    """
    # Load data from database
    df = load_pool_data("defi_prime_rate.db")
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Extract pool 0 APY and weighted APY data
    pool_data = extract_pool_apy_data(df)
    
    if pool_data is None:
        print("Failed to extract pool data. Exiting.")
        return
    
    # Print head and tail
    print("\n=== HEAD (First 10 rows) ===")
    print(pool_data.head(10))
    
    print("\n=== TAIL (Last 10 rows) ===")
    print(pool_data.tail(10))
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(pool_data.describe())
    
    # Print current values
    print("\n=== CURRENT VALUES ===")
    latest_data = pool_data.iloc[-1]
    print(f"Latest Pool 0 APY: {latest_data['Pool_0_APY']:.4f}%")
    print(f"Latest Weighted APY: {latest_data['Weighted_APY']:.4f}%")
    
    # Save to CSV
    csv_filename = "pool_0_apy_data.csv"
    pool_data.to_csv(csv_filename)
    print(f"\nData saved to {csv_filename}")
    print(f"CSV file contains {len(pool_data)} rows of data")

if __name__ == "__main__":
    main()
