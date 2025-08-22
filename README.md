# DeFi Prime Rate Analysis

A comprehensive Python project for analyzing DeFi stablecoin yields and calculating a weighted "Prime Rate" for the DeFi ecosystem.

## Overview

This project fetches data from DeFiLlama and other sources to:
- Calculate a weighted DeFi Prime Rate based on top stablecoin pools by TVL
- Analyze correlations between individual pools and the overall rate
- Compare DeFi yields with traditional market indicators
- Generate academic-style visualizations for research and analysis

## Project Structure

### Core Modules

- **`config.py`** - Centralized configuration with constants, API endpoints, and plotting styles
- **`utils.py`** - Common utility functions for data fetching, database operations, and plotting
- **`spr_fetcher_v1.py`** - Main data fetcher that calculates the DeFi Prime Rate
- **`spr_plotter.py`** - Visualization module for Prime Rate analysis
- **`specific_pools_fetcher.py`** - Fetcher for specific pool analysis
- **`yo_corr.py`** - yoUSD correlation analysis with the Prime Rate
- **`corr_analysis.py`** - Market correlation analysis (BTC, ETH, SPY)
- **`spr_db_csv.py`** - Database export utilities
- **`spr_test.py`** - Testing module for correlation calculations

### Data Files

- **`defi_prime_rate.db`** - SQLite database containing historical pool data
- **`512m_logo.png`** - Logo for plot overlays
- **`.env`** - Environment variables (not included, see setup)

## Setup

### Prerequisites

- Python 3.8+
- Polygon.io API key (for Ethereum price data)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API key:
   ```
   POLYGON_API_KEY=your_polygon_api_key_here
   ```

### Dependencies

- `requests` - API calls
- `pandas` - Data manipulation
- `matplotlib` - Plotting and visualization
- `numpy` - Numerical computations
- `python-dotenv` - Environment variable management
- `Pillow` - Image processing for logo overlays
- `seaborn` - Enhanced plotting styles

## Usage

### 1. Fetch DeFi Prime Rate Data

```bash
python spr_fetcher_v1.py
```

This will:
- Fetch top 100 stablecoin pools by TVL from DeFiLlama
- Calculate weighted average APY (the "DeFi Prime Rate")
- Save data to SQLite database
- Display summary statistics

### 2. Create Prime Rate Visualizations

```bash
python spr_plotter.py
```

Generates:
- Daily vs 14-day moving average trends
- Pool contribution analysis
- Time-series contribution charts

### 3. Analyze Specific Pools

```bash
python specific_pools_fetcher.py
```

Creates visualizations comparing:
- USDC and USDT pool APY trends
- Ethereum price movements

### 4. yoUSD Correlation Analysis

```bash
python yo_corr.py
```

Performs comprehensive analysis of yoUSD vs DeFi Prime Rate:
- Time series comparison
- Rolling correlation and beta
- Scatter plot with trend analysis

### 5. Market Correlation Analysis

```bash
python corr_analysis.py
```

Analyzes correlations between:
- Bitcoin vs S&P 500
- Ethereum vs S&P 500  
- Ethereum vs Bitcoin

### 6. Export Data to CSV

```bash
python spr_db_csv.py
```

Exports specific pool data to CSV format for external analysis.

### 7. Run Tests

```bash
python spr_test.py
```

Runs correlation calculation tests with both synthetic and real market data.

## Key Features

### Data Sources
- **DeFiLlama**: Pool APY and TVL data
- **Polygon.io**: Ethereum and traditional market price data

### Analysis Capabilities
- Weighted average APY calculation based on TVL
- Rolling correlation and beta analysis
- Multi-asset correlation matrices
- Time-series contribution analysis

### Visualization Features
- Academic-style plots with serif fonts
- Consistent color theming
- Logo overlays on all charts
- Multiple chart types (line, scatter, heatmap, stacked area)

## Configuration

### Customizing Analysis Parameters

Edit `config.py` to modify:
- Pool selection criteria
- Rolling window sizes
- Plotting themes and colors
- API endpoints

### Adding New Pools

To analyze additional pools:
1. Add pool IDs to `SPECIFIC_POOL_IDS` in `config.py`
2. Update `POOL_NAMES` mapping
3. Add display names to `DISPLAY_POOL_NAMES` if needed

## Database Schema

### `pool_data` Table
- `date` (index) - Date of observation
- `apy_Pool_N` - APY for pool N
- `tvlUsd_Pool_N` - TVL in USD for pool N  
- `weighted_apy` - Calculated weighted average APY
- `ma_apy_14d` - 14-day moving average APY

### `pool_metadata` Table
- `pool_id` - DeFiLlama pool identifier
- `name` - Pool name
- `current_tvl` - Current TVL in USD
- `current_apy` - Current APY percentage
- `last_updated` - Timestamp of last update

## Contributing

When adding new features:
1. Use the existing `config.py` for constants
2. Add reusable functions to `utils.py`
3. Follow the established type hinting patterns
4. Include comprehensive error handling
5. Add appropriate docstrings

## Notes

- The project uses academic-style plotting with serif fonts and muted colors
- All timestamps are normalized to handle timezone differences
- Rate limiting is implemented for API calls
- The database is purged and refreshed on each run to ensure data freshness

## License

This project is for research and educational purposes.