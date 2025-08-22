"""
DeFi ML Analysis - No Interactive Plots Version

Performs comprehensive ML analysis on DeFi data and saves results to files
without showing interactive plots that might cause timeouts.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Import existing project modules
from config import setup_plotting_style, THEME_PALETTE, MUTED_BLUES
from utils import load_data_from_db

def analyze_defi_with_ml():
    """Perform comprehensive ML analysis on DeFi data."""
    print("=" * 60)
    print("DeFi Machine Learning Analysis")
    print("=" * 60)
    
    # Set up plotting (non-interactive)
    setup_plotting_style()
    plt.ioff()  # Turn off interactive mode
    
    # Load data
    print("Loading DeFi data...")
    pool_data, metadata = load_data_from_db()
    
    if pool_data is None or pool_data.empty:
        print("No data found. Please run the data fetcher first.")
        return
    
    print(f"✓ Loaded data: {pool_data.shape[0]} records, {pool_data.shape[1]} columns")
    print(f"  Date range: {pool_data.index.min()} to {pool_data.index.max()}")
    print(f"  Available pools: {len(metadata)} pools" if metadata is not None else "")
    
    # Create comprehensive features
    print("\nCreating ML features...")
    features_df = create_ml_features(pool_data)
    
    if features_df is None or features_df.empty:
        print("Failed to create features")
        return
    
    print(f"✓ Created {len(features_df)} records with {len(features_df.columns)} features")
    
    # Perform analyses
    results = {}
    
    # 1. Clustering Analysis
    print("\n1. CLUSTERING ANALYSIS")
    print("-" * 30)
    clustering_results = perform_clustering_analysis(features_df)
    results['clustering'] = clustering_results
    
    # 2. Prediction Modeling
    print("\n2. PREDICTION MODELING")
    print("-" * 30)
    prediction_results = build_prediction_models(features_df)
    results['prediction'] = prediction_results
    
    # 3. Anomaly Detection
    print("\n3. ANOMALY DETECTION")
    print("-" * 30)
    anomaly_results = detect_anomalies(features_df)
    results['anomaly'] = anomaly_results
    
    # 4. Statistical Analysis
    print("\n4. STATISTICAL ANALYSIS")
    print("-" * 30)
    stats_results = analyze_statistics(features_df)
    results['statistics'] = stats_results
    
    # 5. Create Visualizations
    print("\n5. CREATING VISUALIZATIONS")
    print("-" * 30)
    create_analysis_plots(features_df, results)
    
    # 6. Generate Summary Report
    print("\n6. GENERATING SUMMARY REPORT")
    print("-" * 30)
    generate_summary_report(results)
    
    print("\n" + "=" * 60)
    print("DeFi ML Analysis Complete!")
    print("✓ All analyses performed successfully")
    print("✓ Visualizations saved as PNG files")
    print("✓ Summary report generated")
    print("=" * 60)

def create_ml_features(pool_data):
    """Create comprehensive ML features from pool data."""
    features_df = pd.DataFrame(index=pool_data.index)
    
    # Market-wide features
    if 'weighted_apy' in pool_data.columns:
        features_df['market_apy'] = pool_data['weighted_apy']
        features_df['market_apy_ma_7d'] = pool_data['weighted_apy'].rolling(7, min_periods=3).mean()
        features_df['market_apy_ma_30d'] = pool_data['weighted_apy'].rolling(30, min_periods=15).mean()
        features_df['market_volatility'] = pool_data['weighted_apy'].rolling(14, min_periods=7).std()
        features_df['market_returns'] = pool_data['weighted_apy'].pct_change()
        features_df['market_trend_7d'] = pool_data['weighted_apy'] - pool_data['weighted_apy'].shift(7)
        features_df['market_momentum'] = (pool_data['weighted_apy'] / 
                                         pool_data['weighted_apy'].rolling(14, min_periods=7).mean() - 1)
    
    # Individual pool features (select pools with sufficient data)
    apy_cols = [col for col in pool_data.columns if col.endswith('_apy')]
    selected_pools = []
    
    for col in apy_cols:
        if pool_data[col].count() > 100:  # At least 100 data points
            pool_name = col.replace('_apy', '')
            
            # Basic APY features
            features_df[f'{pool_name}_apy'] = pool_data[col]
            features_df[f'{pool_name}_volatility'] = pool_data[col].rolling(14, min_periods=7).std()
            features_df[f'{pool_name}_returns'] = pool_data[col].pct_change()
            features_df[f'{pool_name}_ma_7d'] = pool_data[col].rolling(7, min_periods=3).mean()
            
            # Relative features (vs market)
            if 'weighted_apy' in pool_data.columns:
                features_df[f'{pool_name}_vs_market'] = pool_data[col] - pool_data['weighted_apy']
                features_df[f'{pool_name}_beta'] = pool_data[col].rolling(30).corr(pool_data['weighted_apy'])
            
            selected_pools.append(pool_name)
    
    print(f"  Selected {len(selected_pools)} pools with sufficient data")
    
    # Time-based features
    features_df['day_of_week'] = features_df.index.dayofweek
    features_df['month'] = features_df.index.month
    features_df['quarter'] = features_df.index.quarter
    
    # Remove rows with too many missing values
    features_df = features_df.dropna(thresh=len(features_df.columns) * 0.5)
    
    return features_df

def perform_clustering_analysis(features_df):
    """Perform clustering analysis on DeFi pools."""
    # Select APY features for clustering
    apy_cols = [col for col in features_df.columns if '_apy' in col and 'ma_' not in col and 'market' not in col]
    
    if len(apy_cols) < 2:
        print("  Insufficient APY columns for clustering")
        return {}
    
    # Use recent data (last 90 days)
    recent_data = features_df[apy_cols].tail(90).dropna()
    
    if len(recent_data) < 10:
        print("  Insufficient recent data for clustering")
        return {}
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(recent_data)
    
    # Determine optimal number of clusters
    silhouette_scores = []
    K_range = range(2, min(8, len(apy_cols) + 1))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    best_silhouette = max(silhouette_scores)
    
    # Final clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    
    print(f"  ✓ Optimal clusters: {optimal_k}")
    print(f"  ✓ Silhouette score: {best_silhouette:.3f}")
    print(f"  ✓ PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    
    return {
        'optimal_k': optimal_k,
        'silhouette_score': best_silhouette,
        'cluster_labels': cluster_labels,
        'pca_data': pca_data,
        'pca_variance': pca.explained_variance_ratio_,
        'feature_names': apy_cols,
        'recent_data': recent_data
    }

def build_prediction_models(features_df):
    """Build ML models to predict market APY."""
    if 'market_apy' not in features_df.columns:
        print("  No market APY data available for prediction")
        return {}
    
    # Prepare features for prediction
    target_col = 'market_apy'
    feature_cols = [col for col in features_df.columns 
                   if col != target_col and not col.startswith('market_apy')]
    
    # Get clean data
    model_data = features_df[feature_cols + [target_col]].dropna()
    
    if len(model_data) < 50:
        print(f"  Insufficient data for modeling: {len(model_data)} records")
        return {}
    
    X = model_data[feature_cols]
    y = model_data[target_col]
    
    # Split data (time series split)
    split_idx = int(len(model_data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Random Forest (Simple)': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'model': model
        }
        
        print(f"  {name}:")
        print(f"    Train R²: {train_r2:.3f}")
        print(f"    Test R²: {test_r2:.3f}")
        print(f"    Test RMSE: {test_rmse:.3f}")
    
    # Feature importance from best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"  Top 5 Important Features ({best_model_name}):")
        for _, row in importance_df.head().iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")
        
        results['feature_importance'] = importance_df
    
    results['X_test'] = X_test
    results['y_test'] = y_test
    results['best_model'] = best_model_name
    
    return results

def detect_anomalies(features_df):
    """Detect anomalous behavior in DeFi data."""
    # Use market APY and volatility for anomaly detection
    anomaly_cols = ['market_apy', 'market_volatility']
    available_cols = [col for col in anomaly_cols if col in features_df.columns]
    
    if not available_cols:
        print("  No suitable columns for anomaly detection")
        return {}
    
    anomaly_data = features_df[available_cols].dropna()
    
    if len(anomaly_data) < 20:
        print("  Insufficient data for anomaly detection")
        return {}
    
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(anomaly_data)
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_labels = iso_forest.fit_predict(scaled_data)
    
    # Statistical anomaly detection (Z-score method)
    z_scores = np.abs((anomaly_data - anomaly_data.mean()) / anomaly_data.std())
    statistical_anomalies = (z_scores > 2.5).any(axis=1)
    
    n_iso_anomalies = np.sum(anomaly_labels == -1)
    n_stat_anomalies = np.sum(statistical_anomalies)
    
    anomaly_dates = anomaly_data.index[anomaly_labels == -1]
    
    print(f"  ✓ Isolation Forest: {n_iso_anomalies} anomalies ({n_iso_anomalies/len(anomaly_data)*100:.1f}%)")
    print(f"  ✓ Statistical (Z-score): {n_stat_anomalies} anomalies ({n_stat_anomalies/len(anomaly_data)*100:.1f}%)")
    
    if n_iso_anomalies > 0:
        print("  Recent anomaly dates:")
        for date in anomaly_dates[-3:]:
            if 'market_apy' in anomaly_data.columns:
                apy_val = anomaly_data.loc[date, 'market_apy']
                print(f"    {date.strftime('%Y-%m-%d')}: Market APY = {apy_val:.2f}%")
    
    return {
        'isolation_anomalies': anomaly_labels,
        'statistical_anomalies': statistical_anomalies,
        'anomaly_dates': anomaly_dates,
        'n_anomalies': n_iso_anomalies,
        'data': anomaly_data,
        'feature_names': available_cols
    }

def analyze_statistics(features_df):
    """Perform statistical analysis of the data."""
    stats = {}
    
    if 'market_apy' in features_df.columns:
        apy_data = features_df['market_apy'].dropna()
        
        stats['market_apy'] = {
            'mean': apy_data.mean(),
            'std': apy_data.std(),
            'min': apy_data.min(),
            'max': apy_data.max(),
            'current': apy_data.iloc[-1] if len(apy_data) > 0 else None,
            'trend_30d': apy_data.iloc[-1] - apy_data.iloc[-31] if len(apy_data) > 30 else None
        }
        
        print(f"  Market APY Statistics:")
        print(f"    Mean: {stats['market_apy']['mean']:.2f}%")
        print(f"    Std Dev: {stats['market_apy']['std']:.2f}%")
        print(f"    Range: {stats['market_apy']['min']:.2f}% - {stats['market_apy']['max']:.2f}%")
        print(f"    Current: {stats['market_apy']['current']:.2f}%")
        if stats['market_apy']['trend_30d'] is not None:
            trend = stats['market_apy']['trend_30d']
            print(f"    30-day trend: {trend:+.2f}% ({'↑' if trend > 0 else '↓'})")
    
    # Pool correlations
    apy_cols = [col for col in features_df.columns if '_apy' in col and 'ma_' not in col]
    if len(apy_cols) > 1:
        corr_matrix = features_df[apy_cols].corr()
        
        # Find highest and lowest correlations
        corr_values = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_values.append({
                    'pair': f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}",
                    'correlation': corr_matrix.iloc[i, j]
                })
        
        if corr_values:
            corr_df = pd.DataFrame(corr_values).dropna()
            if not corr_df.empty:
                highest_corr = corr_df.loc[corr_df['correlation'].idxmax()]
                lowest_corr = corr_df.loc[corr_df['correlation'].idxmin()]
                
                print(f"  Pool Correlations:")
                print(f"    Highest: {highest_corr['pair'].replace('_apy', '')} = {highest_corr['correlation']:.3f}")
                print(f"    Lowest: {lowest_corr['pair'].replace('_apy', '')} = {lowest_corr['correlation']:.3f}")
                
                stats['correlations'] = {
                    'highest': highest_corr,
                    'lowest': lowest_corr,
                    'matrix': corr_matrix
                }
    
    return stats

def create_analysis_plots(features_df, results):
    """Create and save analysis plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Clustering plot
    if 'clustering' in results and results['clustering']:
        clustering = results['clustering']
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(clustering['pca_data'][:, 0], clustering['pca_data'][:, 1], 
                            c=clustering['cluster_labels'], cmap='tab10', alpha=0.7, s=50)
        plt.xlabel(f'PC1 ({clustering["pca_variance"][0]:.1%} variance)')
        plt.ylabel(f'PC2 ({clustering["pca_variance"][1]:.1%} variance)')
        plt.title(f'DeFi Pool Clustering (k={clustering["optimal_k"]}, Silhouette={clustering["silhouette_score"]:.3f})')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        filename = f'defi_clustering_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=THEME_PALETTE[0])
        print(f"  ✓ Clustering plot saved: {filename}")
        plt.close()
    
    # 2. Prediction plot
    if 'prediction' in results and results['prediction'] and 'X_test' in results['prediction']:
        prediction = results['prediction']
        best_model_name = prediction['best_model']
        best_model = prediction[best_model_name]['model']
        
        # Make predictions for plotting
        X_test_scaled = StandardScaler().fit_transform(prediction['X_test'])
        test_pred = best_model.predict(X_test_scaled)
        y_test = prediction['y_test']
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, test_pred, alpha=0.6, color=MUTED_BLUES[0])
        
        # Perfect prediction line
        min_val = min(y_test.min(), test_pred.min())
        max_val = max(y_test.max(), test_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        plt.xlabel('Actual APY (%)')
        plt.ylabel('Predicted APY (%)')
        plt.title(f'APY Prediction Results - {best_model_name}\n(R² = {prediction[best_model_name]["test_r2"]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f'defi_predictions_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=THEME_PALETTE[0])
        print(f"  ✓ Prediction plot saved: {filename}")
        plt.close()
    
    # 3. Anomaly detection plot
    if 'anomaly' in results and results['anomaly'] and 'market_apy' in features_df.columns:
        anomaly = results['anomaly']
        
        plt.figure(figsize=(12, 6))
        market_data = features_df['market_apy'].dropna()
        plt.plot(market_data.index, market_data.values, color=MUTED_BLUES[0], alpha=0.7, label='Market APY')
        
        if anomaly['n_anomalies'] > 0:
            anomaly_dates = anomaly['anomaly_dates']
            anomaly_values = market_data.loc[anomaly_dates]
            plt.scatter(anomaly_dates, anomaly_values, color='red', s=50, 
                       alpha=0.8, label=f'Anomalies ({anomaly["n_anomalies"]})', zorder=5)
        
        plt.ylabel('APY (%)')
        plt.title('DeFi Market APY - Anomaly Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        filename = f'defi_anomalies_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=THEME_PALETTE[0])
        print(f"  ✓ Anomaly plot saved: {filename}")
        plt.close()
    
    # 4. Feature importance plot
    if 'prediction' in results and 'feature_importance' in results['prediction']:
        importance_df = results['prediction']['feature_importance']
        
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(10)
        plt.barh(range(len(top_features)), top_features['importance'], 
                color=MUTED_BLUES[0], alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Most Important Features for APY Prediction')
        plt.grid(True, alpha=0.3)
        
        filename = f'defi_feature_importance_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=THEME_PALETTE[0])
        print(f"  ✓ Feature importance plot saved: {filename}")
        plt.close()

def generate_summary_report(results):
    """Generate a comprehensive summary report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'defi_ml_report_{timestamp}.txt'
    
    with open(filename, 'w') as f:
        f.write("DeFi Machine Learning Analysis Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Clustering results
        if 'clustering' in results and results['clustering']:
            c = results['clustering']
            f.write("CLUSTERING ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Optimal number of clusters: {c['optimal_k']}\n")
            f.write(f"Silhouette score: {c['silhouette_score']:.3f}\n")
            f.write(f"Features used: {len(c['feature_names'])}\n")
            f.write(f"PCA explained variance: {c['pca_variance'].sum():.1%}\n\n")
        
        # Prediction results
        if 'prediction' in results and results['prediction']:
            p = results['prediction']
            f.write("PREDICTION MODELING\n")
            f.write("-" * 20 + "\n")
            f.write(f"Best model: {p['best_model']}\n")
            f.write(f"Test R² score: {p[p['best_model']]['test_r2']:.3f}\n")
            f.write(f"Test RMSE: {p[p['best_model']]['test_rmse']:.3f}\n")
            
            if 'feature_importance' in p:
                f.write("\nTop 5 Important Features:\n")
                for _, row in p['feature_importance'].head().iterrows():
                    f.write(f"  {row['feature']}: {row['importance']:.3f}\n")
            f.write("\n")
        
        # Anomaly results
        if 'anomaly' in results and results['anomaly']:
            a = results['anomaly']
            f.write("ANOMALY DETECTION\n")
            f.write("-" * 18 + "\n")
            f.write(f"Anomalies detected: {a['n_anomalies']}\n")
            f.write(f"Percentage of data: {a['n_anomalies']/len(a['data'])*100:.1f}%\n")
            if a['n_anomalies'] > 0:
                f.write("Recent anomaly dates:\n")
                for date in a['anomaly_dates'][-3:]:
                    f.write(f"  {date.strftime('%Y-%m-%d')}\n")
            f.write("\n")
        
        # Statistics
        if 'statistics' in results and results['statistics']:
            s = results['statistics']
            if 'market_apy' in s:
                apy = s['market_apy']
                f.write("MARKET STATISTICS\n")
                f.write("-" * 17 + "\n")
                f.write(f"Mean APY: {apy['mean']:.2f}%\n")
                f.write(f"Std Dev: {apy['std']:.2f}%\n")
                f.write(f"Range: {apy['min']:.2f}% - {apy['max']:.2f}%\n")
                f.write(f"Current APY: {apy['current']:.2f}%\n")
                if apy['trend_30d'] is not None:
                    f.write(f"30-day trend: {apy['trend_30d']:+.2f}%\n")
                f.write("\n")
            
            if 'correlations' in s:
                f.write("POOL CORRELATIONS\n")
                f.write("-" * 16 + "\n")
                f.write(f"Highest correlation: {s['correlations']['highest']['correlation']:.3f}\n")
                f.write(f"Lowest correlation: {s['correlations']['lowest']['correlation']:.3f}\n")
        
        f.write("\nAnalysis completed successfully.\n")
    
    print(f"  ✓ Summary report saved: {filename}")

if __name__ == "__main__":
    analyze_defi_with_ml()