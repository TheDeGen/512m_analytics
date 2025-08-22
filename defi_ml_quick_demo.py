"""
Quick DeFi ML Analysis Demo

A streamlined version of the ML analysis that demonstrates key capabilities
without complex visualizations that might timeout.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.decomposition import PCA

# Import existing project modules
from config import setup_plotting_style, THEME_PALETTE, MUTED_BLUES
from utils import load_data_from_db, print_data_summary

def quick_ml_demo():
    """Run a quick ML demonstration on DeFi data."""
    print("=" * 50)
    print("DeFi ML Quick Demo")
    print("=" * 50)
    
    # Set up plotting
    setup_plotting_style()
    
    # Load data
    print("Loading DeFi data...")
    pool_data, metadata = load_data_from_db()
    
    if pool_data is None or pool_data.empty:
        print("No data found. Please run the data fetcher first.")
        return
    
    print(f"Loaded data: {pool_data.shape[0]} records, {pool_data.shape[1]} columns")
    print(f"Date range: {pool_data.index.min()} to {pool_data.index.max()}")
    
    # Create simple features
    features_df = pd.DataFrame(index=pool_data.index)
    
    # Market APY features
    if 'weighted_apy' in pool_data.columns:
        features_df['market_apy'] = pool_data['weighted_apy']
        features_df['market_apy_ma_7d'] = pool_data['weighted_apy'].rolling(7).mean()
        features_df['market_apy_ma_30d'] = pool_data['weighted_apy'].rolling(30).mean()
        features_df['market_volatility'] = pool_data['weighted_apy'].rolling(14).std()
        features_df['market_returns'] = pool_data['weighted_apy'].pct_change()
    
    # Individual pool features (top 3 pools by data availability)
    apy_cols = [col for col in pool_data.columns if col.endswith('_apy')]
    top_pools = []
    for col in apy_cols[:3]:  # Take first 3 pools
        if pool_data[col].count() > 100:  # Ensure sufficient data
            pool_name = col.replace('_apy', '')
            features_df[f'{pool_name}_apy'] = pool_data[col]
            features_df[f'{pool_name}_volatility'] = pool_data[col].rolling(14).std()
            top_pools.append(pool_name)
    
    # Remove rows with too many NaN values
    features_df = features_df.dropna(thresh=len(features_df.columns) * 0.6)
    
    print(f"Created {len(features_df)} feature records with {len(features_df.columns)} features")
    
    if len(features_df) < 50:
        print("Insufficient data for ML analysis")
        return
    
    # 1. Clustering Analysis
    print("\n1. Clustering Analysis")
    print("-" * 20)
    
    # Use APY columns for clustering
    apy_features = [col for col in features_df.columns if 'apy' in col and 'ma_' not in col]
    cluster_data = features_df[apy_features].tail(60).dropna()  # Last 60 days
    
    if not cluster_data.empty and len(cluster_data) > 10:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Clusters found: {len(np.unique(cluster_labels))}")
        
        # PCA for visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, 
                            cmap='tab10', alpha=0.7, s=50)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title(f'DeFi Pool Clustering (Silhouette: {silhouette_avg:.3f})')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'defi_clustering_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"Clustering plot saved as defi_clustering_{timestamp}.png")
        plt.show()
    
    # 2. Prediction Model
    print("\n2. APY Prediction Model")
    print("-" * 25)
    
    if 'market_apy' in features_df.columns:
        # Prepare data for prediction
        feature_cols = [col for col in features_df.columns 
                       if col != 'market_apy' and 'market_apy' not in col]
        
        model_data = features_df[feature_cols + ['market_apy']].dropna()
        
        if len(model_data) > 30:
            X = model_data[feature_cols]
            y = model_data['market_apy']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
            rf_model.fit(X_train_scaled, y_train)
            
            # Make predictions
            train_pred = rf_model.predict(X_train_scaled)
            test_pred = rf_model.predict(X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            print(f"Random Forest Results:")
            print(f"  Training R²: {train_r2:.3f}")
            print(f"  Test R²: {test_r2:.3f}")
            print(f"  Test RMSE: {test_rmse:.3f}")
            
            # Feature importance
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 5 Most Important Features:")
            for i, row in importance_df.head().iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
            
            # Plot predictions vs actual
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, test_pred, alpha=0.6, color=MUTED_BLUES[0])
            
            # Perfect prediction line
            min_val = min(y_test.min(), test_pred.min())
            max_val = max(y_test.max(), test_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, 
                    label='Perfect Prediction')
            
            plt.xlabel('Actual APY (%)')
            plt.ylabel('Predicted APY (%)')
            plt.title(f'APY Prediction Results (R² = {test_r2:.3f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'defi_predictions_{timestamp}.png', dpi=300, bbox_inches='tight')
            print(f"Prediction plot saved as defi_predictions_{timestamp}.png")
            plt.show()
    
    # 3. Anomaly Detection
    print("\n3. Anomaly Detection")
    print("-" * 20)
    
    # Use market APY and volatility for anomaly detection
    anomaly_features = ['market_apy', 'market_volatility']
    anomaly_data = features_df[anomaly_features].dropna()
    
    if not anomaly_data.empty and len(anomaly_data) > 20:
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(anomaly_data)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(scaled_data)
        
        n_anomalies = np.sum(anomaly_labels == -1)
        anomaly_dates = anomaly_data.index[anomaly_labels == -1]
        
        print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(anomaly_data)*100:.1f}% of data)")
        
        if n_anomalies > 0:
            print("Anomaly dates:")
            for date in anomaly_dates[-5:]:  # Show last 5 anomalies
                apy_val = anomaly_data.loc[date, 'market_apy']
                print(f"  {date.strftime('%Y-%m-%d')}: APY = {apy_val:.2f}%")
        
        # Plot anomalies
        plt.figure(figsize=(12, 6))
        plt.plot(anomaly_data.index, anomaly_data['market_apy'], 
                color=MUTED_BLUES[0], alpha=0.7, label='Market APY')
        
        if n_anomalies > 0:
            anomaly_values = anomaly_data.loc[anomaly_dates, 'market_apy']
            plt.scatter(anomaly_dates, anomaly_values, color='red', s=50, 
                       alpha=0.8, label='Anomalies', zorder=5)
        
        plt.ylabel('APY (%)')
        plt.title('DeFi Market APY - Anomaly Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'defi_anomalies_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"Anomaly plot saved as defi_anomalies_{timestamp}.png")
        plt.show()
    
    # 4. Summary Statistics
    print("\n4. Summary Statistics")
    print("-" * 20)
    
    if 'market_apy' in features_df.columns:
        apy_data = features_df['market_apy'].dropna()
        print(f"Market APY Statistics:")
        print(f"  Mean: {apy_data.mean():.2f}%")
        print(f"  Std Dev: {apy_data.std():.2f}%")
        print(f"  Min: {apy_data.min():.2f}%")
        print(f"  Max: {apy_data.max():.2f}%")
        print(f"  Current: {apy_data.iloc[-1]:.2f}%")
    
    print("\n" + "=" * 50)
    print("DeFi ML Quick Demo Complete!")
    print("Generated plots show:")
    print("✓ Pool clustering patterns")
    print("✓ APY prediction accuracy") 
    print("✓ Anomaly detection results")
    print("=" * 50)

if __name__ == "__main__":
    quick_ml_demo()