import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime, timedelta
from dotenv import load_dotenv
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Load environment variables
load_dotenv()

# Theme palette (from spr_plotter.py)
theme_palette = ['#f7f3ec', '#ede4da', '#b9a58f', '#574c40', '#36312a']
muted_blues = [
    '#2b3e50', '#3c5a77', '#4f7192', '#5f86a8', '#6f9bbd',
    '#86abc7', '#9bbad1', '#afc8da', '#c3d5e3', '#d7e2ec'
]

# Set academic-style plotting with serif fonts and beige background (matching spr_plotter.py)
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    # Background from palette (first color)
    'figure.facecolor': theme_palette[0],
    'axes.facecolor': theme_palette[0],
    'savefig.facecolor': theme_palette[0]
})

def add_logo_overlay(ax, logo_path="512m_logo.png", alpha=0.05):
    """
    Add logo overlay to the center of the plot
    
    Args:
        ax: matplotlib axis object
        logo_path (str): path to logo image
        alpha (float): transparency level (0-1)
    """
    try:
        # Load the logo image
        logo_img = Image.open(logo_path)
        
        # Convert to numpy array and normalize
        logo_array = np.array(logo_img)
        
        # Get the center of the plot
        x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
        y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
        
        # Calculate appropriate size for the logo (about 28% of plot width - 30% smaller than before)
        plot_width = ax.get_xlim()[1] - ax.get_xlim()[0]
        logo_width = plot_width * 0.2
        
        # Create offset image
        im = OffsetImage(logo_array, zoom=logo_width/logo_img.width, alpha=alpha)
        
        # Create annotation box at center
        ab = AnnotationBbox(im, (x_center, y_center), frameon=False)
        ax.add_artist(ab)
        
    except Exception as e:
        print(f"Warning: Could not add logo overlay: {e}")

def fetch_polygon_data(symbol, start_date, end_date, api_key):
    """Fetch data from Polygon.io API"""
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
        print(f"Fetching data for {symbol}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('results'):
            df = pd.DataFrame(data['results'])
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df = df[['date', 'c']].rename(columns={'c': 'close'})
            df.set_index('date', inplace=True)
            
            print(f"Successfully fetched {len(df)} data points for {symbol}")
            return df
        else:
            print(f"No data returned for {symbol}")
            return None
            
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def main():
    """Main function to perform correlation analysis"""
    # Get API key from environment
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("Error: POLYGON_API_KEY environment variable not set")
        return
    
    # Download data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=600)  # Download 600 days of data
    
    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Get Bitcoin, Ethereum, and SPY data
    btc_data = fetch_polygon_data('BTC', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), api_key)
    eth_data = fetch_polygon_data('ETH', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), api_key)
    spy_data = fetch_polygon_data('SPY', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), api_key)
    
    if btc_data is None or eth_data is None or spy_data is None:
        print("Failed to fetch data. Exiting.")
        return
    
    # Debug: Print raw data info
    print(f"\n=== Raw Data Info ===")
    print(f"BTC data shape: {btc_data.shape}")
    print(f"BTC date range: {btc_data.index.min()} to {btc_data.index.max()}")
    print(f"BTC first 5 rows:")
    print(btc_data.head())
    
    print(f"\nETH data shape: {eth_data.shape}")
    print(f"ETH date range: {eth_data.index.min()} to {eth_data.index.max()}")
    
    print(f"\nSPY data shape: {spy_data.shape}")
    print(f"SPY date range: {spy_data.index.min()} to {spy_data.index.max()}")
    print(f"SPY first 5 rows:")
    print(spy_data.head())
    
    # Normalize timestamps to midnight to handle time component differences
    print(f"\n=== Normalizing Timestamps ===")
    
    # Convert all dates to date-only (remove time component)
    btc_data.index = btc_data.index.normalize()
    eth_data.index = eth_data.index.normalize()
    spy_data.index = spy_data.index.normalize()
    
    print(f"After normalization:")
    print(f"BTC first 5 dates: {btc_data.index[:5]}")
    print(f"SPY first 5 dates: {spy_data.index[:5]}")
    
    # Find the common date range (now using normalized dates)
    common_start = max(btc_data.index.min(), eth_data.index.min(), spy_data.index.min())
    common_end = min(btc_data.index.max(), eth_data.index.max(), spy_data.index.max())
    
    print(f"\n=== Common Date Range (Normalized) ===")
    print(f"Common start: {common_start}")
    print(f"Common end: {common_end}")
    
    # Filter all datasets to common date range
    btc_filtered = btc_data[common_start:common_end]
    eth_filtered = eth_data[common_start:common_end]
    spy_filtered = spy_data[common_start:common_end]
    
    print(f"\n=== Filtered Data Info ===")
    print(f"BTC filtered shape: {btc_filtered.shape}")
    print(f"ETH filtered shape: {eth_filtered.shape}")
    print(f"SPY filtered shape: {spy_filtered.shape}")
    
    # Create a DataFrame with all assets (simple alignment)
    df = pd.DataFrame(index=btc_filtered.index)
    df['BTC'] = btc_filtered['close']
    df['ETH'] = eth_filtered['close']
    df['SPY'] = spy_filtered['close']
    
    # Forward fill any missing values (handles trading calendar differences)
    df = df.ffill().dropna()
    
    print(f"\n=== Final Combined Data ===")
    print(f"Combined data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"First 5 rows:")
    print(df.head())
    print(f"Last 5 rows:")
    print(df.tail())
    
    if df.shape[0] == 0:
        print("Error: No data after alignment. Exiting.")
        return
    
    # Calculate returns
    returns = df.pct_change().dropna()
    print(f"\nReturns data shape: {returns.shape}")
    
    # Create windows for correlation calculation
    windows = range(14, 181, 14)  # From 14 to 180 days, step 14
    
    # Calculate rolling correlations for each window using full dataset
    btc_corr_matrix = np.zeros((len(windows), len(returns)))
    eth_corr_matrix = np.zeros((len(windows), len(returns)))
    eth_spy_corr_matrix = np.zeros((len(windows), len(returns)))
    
    for i, window in enumerate(windows):
        btc_corr_matrix[i] = returns['BTC'].rolling(window).corr(returns['SPY'])
        eth_corr_matrix[i] = returns['ETH'].rolling(window).corr(returns['BTC']) # Changed to BTC
        eth_spy_corr_matrix[i] = returns['ETH'].rolling(window).corr(returns['SPY'])
    
    # Select only the last 360 days for plotting
    plot_start_date = end_date - timedelta(days=360)
    plot_mask = returns.index >= plot_start_date
    btc_corr_matrix = btc_corr_matrix[:, plot_mask]
    eth_corr_matrix = eth_corr_matrix[:, plot_mask]
    eth_spy_corr_matrix = eth_spy_corr_matrix[:, plot_mask]
    dates = returns.index[plot_mask]
    
    print(f"\n=== Plotting Data ===")
    print(f"Plot data shape: {btc_corr_matrix.shape}")
    print(f"Number of dates for plotting: {len(dates)}")
    
    if len(dates) == 0:
        print("Error: No dates for plotting. Exiting.")
        return
    
    # Convert dates to numeric values for plotting
    date_nums = np.array([d.toordinal() for d in dates])
    
    # Define indices for axis formatting
    n_dates = min(5, len(dates))
    if n_dates > 1:
        step = max(1, len(dates) // (n_dates - 1))
        selected_indices = list(range(0, len(dates), step))
        if len(selected_indices) < n_dates:
            selected_indices.append(len(dates) - 1)
    else:
        selected_indices = [0]
    
    n_windows = min(5, len(windows))
    if n_windows > 1:
        window_step = max(1, len(windows) // (n_windows - 1))
        selected_windows = list(range(0, len(windows), window_step))
        if len(selected_windows) < n_windows:
            selected_windows.append(len(windows) - 1)
    else:
        selected_windows = [0]
    
    # Create second figure with combined heatmap
    fig2, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(18, 5))
    
    # First subplot - Bitcoin correlation heatmap
    im2 = ax2.imshow(btc_corr_matrix, cmap='Pastel1', aspect='auto', 
                      extent=[date_nums[0], date_nums[-1], windows[0], windows[-1]], 
                      origin='lower')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Window Size (days)')
    ax2.set_title('Bitcoin-SPY Correlation')
    
    # Format axes for Bitcoin plot
    ax2.set_xticks([date_nums[i] for i in selected_indices])
    ax2.set_xticklabels([dates[i].strftime('%m-%d') for i in selected_indices], rotation=45)
    ax2.set_yticks([windows[i] for i in selected_windows])
    
    # Add colorbar for Bitcoin
    cbar2 = fig2.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Correlation')
    
    # Second subplot - Ethereum-SPY correlation heatmap
    im3 = ax3.imshow(eth_spy_corr_matrix, cmap='Pastel1', aspect='auto', 
                      extent=[date_nums[0], date_nums[-1], windows[0], windows[-1]], 
                      origin='lower')
    
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Window Size (days)')
    ax3.set_title('Ethereum-SPY Correlation')
    
    # Format axes for Ethereum-SPY plot
    ax3.set_xticks([date_nums[i] for i in selected_indices])
    ax3.set_xticklabels([dates[i].strftime('%m-%d') for i in selected_indices], rotation=45)
    ax3.set_yticks([windows[i] for i in selected_windows])
    
    # Add colorbar for Ethereum-SPY
    cbar3 = fig2.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Correlation')
    
    # Third subplot - Ethereum-Bitcoin correlation heatmap
    im4 = ax4.imshow(eth_corr_matrix, cmap='Pastel1', aspect='auto', 
                      extent=[date_nums[0], date_nums[-1], windows[0], windows[-1]], 
                      origin='lower')
    
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Window Size (days)')
    ax4.set_title('Ethereum-Bitcoin Correlation')
    
    # Format axes for Ethereum-Bitcoin plot
    ax4.set_xticks([date_nums[i] for i in selected_indices])
    ax4.set_xticklabels([dates[i].strftime('%m-%d') for i in selected_indices], rotation=45)
    ax4.set_yticks([windows[i] for i in selected_windows])
    
    # Add colorbar for Ethereum-Bitcoin
    cbar4 = fig2.colorbar(im4, ax=ax4, shrink=0.8)
    cbar4.set_label('Correlation')
    
    # Add logo overlays to heatmap figure
    add_logo_overlay(ax2)
    add_logo_overlay(ax3)
    add_logo_overlay(ax4)
    
    plt.tight_layout()
    plt.show()
    
    # Create third figure with line plots for specific window sizes
    fig3, (ax5, ax6, ax7) = plt.subplots(3, 1, figsize=(12, 10))
    
    # First subplot - BTC-SPY betas
    # Calculate 30-day and 90-day betas with division by zero protection
    spy_var_30d = returns['SPY'].rolling(window=30).var()
    spy_var_90d = returns['SPY'].rolling(window=90).var()
    
    btc_spy_30d = returns['BTC'].rolling(window=30).cov(returns['SPY']) / spy_var_30d.where(spy_var_30d > 1e-10, np.nan)
    btc_spy_90d = returns['BTC'].rolling(window=90).cov(returns['SPY']) / spy_var_90d.where(spy_var_90d > 1e-10, np.nan)
    
    ax5.plot(btc_spy_30d.index, btc_spy_30d, label='30-day', linewidth=2, color=muted_blues[0], alpha=0.2)
    ax5.plot(btc_spy_90d.index, btc_spy_90d, label='90-day', linewidth=2, color=muted_blues[2])
    ax5.axhline(y=0, color=theme_palette[3], linestyle='--', alpha=0.7, linewidth=1)
    ax5.axhline(y=1, color=theme_palette[2], linestyle=':', alpha=0.5, linewidth=1, label='Beta = 1')
    
    ax5.set_title('Bitcoin-SPY Beta')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Beta Coefficient')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    
    # Second subplot - ETH-SPY betas
    eth_spy_30d = returns['ETH'].rolling(window=30).cov(returns['SPY']) / spy_var_30d.where(spy_var_30d > 1e-10, np.nan)
    eth_spy_90d = returns['ETH'].rolling(window=90).cov(returns['SPY']) / spy_var_90d.where(spy_var_90d > 1e-10, np.nan)
    
    ax6.plot(eth_spy_30d.index, eth_spy_30d, label='30-day', linewidth=2, color=muted_blues[0], alpha=0.2)
    ax6.plot(eth_spy_90d.index, eth_spy_90d, label='90-day', linewidth=2, color=muted_blues[2])
    ax6.axhline(y=0, color=theme_palette[3], linestyle='--', alpha=0.7, linewidth=1)
    ax6.axhline(y=1, color=theme_palette[2], linestyle=':', alpha=0.5, linewidth=1, label='Beta = 1')
    
    ax6.set_title('Ethereum-SPY Beta')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Beta Coefficient')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Third subplot - ETH-BTC betas
    btc_var_30d = returns['BTC'].rolling(window=30).var()
    btc_var_90d = returns['BTC'].rolling(window=90).var()
    
    eth_btc_30d = returns['ETH'].rolling(window=30).cov(returns['BTC']) / btc_var_30d.where(btc_var_30d > 1e-10, np.nan)
    eth_btc_90d = returns['ETH'].rolling(window=90).cov(returns['BTC']) / btc_var_90d.where(btc_var_90d > 1e-10, np.nan)
    
    ax7.plot(eth_btc_30d.index, eth_btc_30d, label='30-day', linewidth=2, color=muted_blues[0], alpha=0.2)
    ax7.plot(eth_btc_90d.index, eth_btc_90d, label='90-day', linewidth=2, color=muted_blues[2])
    ax7.axhline(y=0, color=theme_palette[3], linestyle='--', alpha=0.7, linewidth=1)
    ax7.axhline(y=1, color=theme_palette[2], linestyle=':', alpha=0.5, linewidth=1, label='Beta = 1')
    
    ax7.set_title('Ethereum-Bitcoin Beta')
    ax7.set_xlabel('Date')
    ax7.set_ylabel('Beta Coefficient')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Format x-axis dates for all subplots
    for ax in [ax5, ax6, ax7]:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add logo overlays to line plot figure
    add_logo_overlay(ax5)
    add_logo_overlay(ax6)
    add_logo_overlay(ax7)
    
    plt.tight_layout()
    plt.show()
    
    print("Correlation analysis completed!")

if __name__ == "__main__":
    main()
