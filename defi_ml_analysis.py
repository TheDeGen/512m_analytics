"""
DeFi Machine Learning Analysis Module

This module uses existing APIs and utility functions to pull DeFi data, 
analyze interesting relationships using machine learning techniques, and 
create comprehensive visualizations including:
- Time series forecasting for APY prediction
- Clustering analysis of DeFi pools
- Volatility prediction models  
- Correlation network analysis
- Risk-return optimization
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Import existing project modules
from config import (
    API_ENDPOINTS, DEFAULT_DB_FILENAME, THEME_PALETTE, MUTED_BLUES,
    setup_plotting_style, ROLLING_WINDOW_SIZES
)
from utils import (
    fetch_pool_chart_data, load_data_from_db, add_logo_overlay,
    format_date_axis, validate_dataframe, normalize_datetime_index,
    print_data_summary, create_subplot_grid, safe_api_request
)
from spr_fetcher_v1 import fetch_top_stablecoin_pools_by_tvl


class DeFiMLAnalyzer:
    """
    Machine Learning analyzer for DeFi data with comprehensive analysis capabilities.
    """
    
    def __init__(self, db_filename: str = DEFAULT_DB_FILENAME):
        """Initialize the analyzer with database connection."""
        self.db_filename = db_filename
        self.pool_data = None
        self.metadata = None
        self.features_df = None
        self.scaler = StandardScaler()
        
        # Set up plotting style
        setup_plotting_style()
        
    def load_and_prepare_data(self) -> bool:
        """
        Load data from database and prepare features for ML analysis.
        
        Returns:
            True if successful, False otherwise
        """
        print("Loading and preparing DeFi data for ML analysis...")
        
        # Load data from database
        self.pool_data, self.metadata = load_data_from_db(self.db_filename)
        
        if self.pool_data is None or self.pool_data.empty:
            print("No data found in database. Please run data fetcher first.")
            return False
        
        # Create comprehensive features
        self.features_df = self._create_features()
        
        if self.features_df is None or self.features_df.empty:
            print("Failed to create features from data")
            return False
        
        print(f"Successfully prepared {len(self.features_df)} data points with {len(self.features_df.columns)} features")
        return True
    
    def _create_features(self) -> Optional[pd.DataFrame]:
        """
        Create comprehensive features from raw pool data for ML analysis.
        
        Returns:
            DataFrame with engineered features
        """
        try:
            df = self.pool_data.copy()
            features = pd.DataFrame(index=df.index)
            
            # Basic features from each pool column
            for col in df.columns:
                if col.endswith('_apy'):
                    pool_name = col.replace('_apy', '')
                    
                    # Current APY
                    features[f'{pool_name}_apy'] = df[col]
                    
                    # Rolling statistics
                    features[f'{pool_name}_apy_ma_7d'] = df[col].rolling(7, min_periods=3).mean()
                    features[f'{pool_name}_apy_ma_30d'] = df[col].rolling(30, min_periods=15).mean()
                    features[f'{pool_name}_apy_std_30d'] = df[col].rolling(30, min_periods=15).std()
                    
                    # Volatility measures
                    features[f'{pool_name}_volatility'] = df[col].rolling(14, min_periods=7).std()
                    features[f'{pool_name}_returns'] = df[col].pct_change()
                    
                    # Trend features
                    features[f'{pool_name}_trend_7d'] = df[col] - df[col].shift(7)
                    features[f'{pool_name}_trend_30d'] = df[col] - df[col].shift(30)
                    
                    # Momentum indicators
                    features[f'{pool_name}_momentum'] = df[col] / df[col].rolling(14, min_periods=7).mean() - 1
                
                elif col.endswith('_tvl'):
                    pool_name = col.replace('_tvl', '')
                    
                    # TVL features
                    features[f'{pool_name}_tvl'] = df[col]
                    features[f'{pool_name}_tvl_ma_7d'] = df[col].rolling(7, min_periods=3).mean()
                    features[f'{pool_name}_tvl_change'] = df[col].pct_change()
            
            # Market-wide features
            if 'weighted_apy' in df.columns:
                features['market_apy'] = df['weighted_apy']
                features['market_apy_ma_30d'] = df['weighted_apy'].rolling(30, min_periods=15).mean()
                features['market_volatility'] = df['weighted_apy'].rolling(14, min_periods=7).std()
                features['market_trend'] = df['weighted_apy'] - df['weighted_apy'].shift(30)
            
            # Time-based features
            features['day_of_week'] = features.index.dayofweek
            features['month'] = features.index.month
            features['quarter'] = features.index.quarter
            
            # Remove rows with too many NaN values
            features = features.dropna(thresh=len(features.columns) * 0.5)
            
            return features
            
        except Exception as e:
            print(f"Error creating features: {e}")
            return None
    
    def perform_clustering_analysis(self, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Perform clustering analysis on DeFi pools to identify similar behavior patterns.
        
        Args:
            n_clusters: Number of clusters for K-means
            
        Returns:
            Dictionary containing clustering results and metrics
        """
        print("Performing clustering analysis on DeFi pools...")
        
        # Prepare data for clustering - use APY and volatility features
        apy_cols = [col for col in self.features_df.columns if col.endswith('_apy') and not 'ma_' in col]
        vol_cols = [col for col in self.features_df.columns if 'volatility' in col]
        cluster_features = apy_cols + vol_cols
        
        # Get recent data for clustering (last 90 days)
        recent_data = self.features_df[cluster_features].tail(90).dropna()
        
        if recent_data.empty:
            print("No suitable data for clustering")
            return {}
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(recent_data)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        
        # DBSCAN for density-based clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(scaled_data)
        
        # PCA for dimensionality reduction and visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        results = {
            'kmeans_labels': cluster_labels,
            'dbscan_labels': dbscan_labels,
            'pca_data': pca_data,
            'silhouette_score': silhouette_avg,
            'cluster_centers': kmeans.cluster_centers_,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'feature_names': cluster_features,
            'recent_data': recent_data
        }
        
        print(f"Clustering complete. Silhouette score: {silhouette_avg:.3f}")
        return results
    
    def build_apy_prediction_model(self, target_pool: str = None) -> Dict[str, Any]:
        """
        Build machine learning models to predict APY movements.
        
        Args:
            target_pool: Specific pool to predict (if None, uses market APY)
            
        Returns:
            Dictionary containing model results and predictions
        """
        print("Building APY prediction models...")
        
        # Determine target variable
        if target_pool and f'{target_pool}_apy' in self.features_df.columns:
            target_col = f'{target_pool}_apy'
            print(f"Predicting APY for {target_pool}")
        elif 'market_apy' in self.features_df.columns:
            target_col = 'market_apy'
            print("Predicting market-wide APY")
        else:
            print("No suitable target variable found for prediction")
            return {}
        
        # Prepare features and target
        feature_cols = [col for col in self.features_df.columns 
                       if col != target_col and not col.startswith(target_col.replace('_apy', ''))]
        
        # Get data with no missing values
        model_data = self.features_df[feature_cols + [target_col]].dropna()
        
        if len(model_data) < 50:
            print("Insufficient data for model training")
            return {}
        
        X = model_data[feature_cols]
        y = model_data[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_pred': train_pred,
                'test_pred': test_pred
            }
            
            print(f"{name} - Test R²: {test_r2:.3f}, Test RMSE: {test_rmse:.3f}")
        
        # Feature importance for Random Forest
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = feature_importance
        
        results['target_col'] = target_col
        results['X_train'] = X_train
        results['X_test'] = X_test
        results['y_train'] = y_train
        results['y_test'] = y_test
        
        return results
    
    def detect_anomalies(self) -> Dict[str, Any]:
        """
        Detect anomalous behavior in DeFi pools using Isolation Forest.
        
        Returns:
            Dictionary containing anomaly detection results
        """
        print("Detecting anomalies in DeFi data...")
        
        # Use APY and volatility features for anomaly detection
        apy_cols = [col for col in self.features_df.columns if '_apy' in col and 'ma_' not in col]
        vol_cols = [col for col in self.features_df.columns if 'volatility' in col]
        anomaly_features = apy_cols + vol_cols
        
        data = self.features_df[anomaly_features].dropna()
        
        if data.empty:
            print("No suitable data for anomaly detection")
            return {}
        
        # Scale the data
        scaled_data = StandardScaler().fit_transform(data)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(scaled_data)
        anomaly_scores = iso_forest.decision_function(scaled_data)
        
        # Statistical anomaly detection (Z-score)
        z_scores = np.abs(stats.zscore(scaled_data, axis=0))
        statistical_anomalies = (z_scores > 3).any(axis=1)
        
        results = {
            'isolation_forest_labels': anomaly_labels,
            'anomaly_scores': anomaly_scores,
            'statistical_anomalies': statistical_anomalies,
            'anomaly_dates': data.index[anomaly_labels == -1],
            'feature_names': anomaly_features,
            'data': data
        }
        
        n_anomalies = np.sum(anomaly_labels == -1)
        print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(data)*100:.1f}% of data)")
        
        return results
    
    def create_comprehensive_plots(self, clustering_results: Dict, prediction_results: Dict, 
                                 anomaly_results: Dict) -> None:
        """
        Create comprehensive visualization plots for all ML analyses.
        
        Args:
            clustering_results: Results from clustering analysis
            prediction_results: Results from prediction models
            anomaly_results: Results from anomaly detection
        """
        print("Creating comprehensive ML analysis plots...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Clustering visualization
        if clustering_results:
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_clustering(ax1, clustering_results)
            
        # 2. Feature importance
        if prediction_results and 'feature_importance' in prediction_results:
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_feature_importance(ax2, prediction_results['feature_importance'])
        
        # 3. Model performance comparison
        if prediction_results:
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_model_comparison(ax3, prediction_results)
        
        # 4. Prediction vs Actual
        if prediction_results:
            ax4 = fig.add_subplot(gs[1, :])
            self._plot_predictions(ax4, prediction_results)
        
        # 5. Anomaly detection
        if anomaly_results:
            ax5 = fig.add_subplot(gs[2, 0])
            self._plot_anomalies(ax5, anomaly_results)
        
        # 6. Correlation heatmap
        ax6 = fig.add_subplot(gs[2, 1:])
        self._plot_correlation_heatmap(ax6)
        
        # 7. Time series analysis
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_time_series_analysis(ax7)
        
        # Add main title and logo
        fig.suptitle('DeFi Machine Learning Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add logo to the figure
        try:
            add_logo_overlay(ax1)
        except:
            pass
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'defi_ml_analysis_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=THEME_PALETTE[0])
        print(f"Comprehensive analysis plot saved as {filename}")
        
        plt.show()
    
    def _plot_clustering(self, ax, results):
        """Plot clustering results."""
        pca_data = results['pca_data']
        labels = results['kmeans_labels']
        
        scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, 
                           cmap='tab10', alpha=0.7, s=50)
        ax.set_xlabel(f'PC1 ({results["pca_explained_variance"][0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({results["pca_explained_variance"][1]:.1%} variance)')
        ax.set_title(f'Pool Clustering (Silhouette: {results["silhouette_score"]:.3f})')
        ax.grid(True, alpha=0.3)
    
    def _plot_feature_importance(self, ax, importance_df):
        """Plot feature importance from Random Forest."""
        top_features = importance_df.head(10)
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      color=MUTED_BLUES[0], alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=8)
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importance')
        ax.grid(True, alpha=0.3)
    
    def _plot_model_comparison(self, ax, results):
        """Plot model performance comparison."""
        models = [name for name in results.keys() if isinstance(results[name], dict)]
        test_r2 = [results[model]['test_r2'] for model in models]
        
        bars = ax.bar(models, test_r2, color=MUTED_BLUES[:len(models)], alpha=0.7)
        ax.set_ylabel('Test R² Score')
        ax.set_title('Model Performance Comparison')
        ax.set_ylim(0, 1)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, test_r2):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    def _plot_predictions(self, ax, results):
        """Plot predictions vs actual values."""
        best_model = max(results.keys(), 
                        key=lambda x: results[x]['test_r2'] if isinstance(results[x], dict) else -1)
        
        if isinstance(results[best_model], dict):
            y_test = results['y_test']
            test_pred = results[best_model]['test_pred']
            
            ax.scatter(y_test, test_pred, alpha=0.6, color=MUTED_BLUES[0])
            
            # Perfect prediction line
            min_val = min(y_test.min(), test_pred.min())
            max_val = max(y_test.max(), test_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel('Actual APY (%)')
            ax.set_ylabel('Predicted APY (%)')
            ax.set_title(f'{best_model} - Predictions vs Actual (R² = {results[best_model]["test_r2"]:.3f})')
            ax.grid(True, alpha=0.3)
    
    def _plot_anomalies(self, ax, results):
        """Plot anomaly detection results."""
        data = results['data']
        anomaly_labels = results['isolation_forest_labels']
        
        # Plot time series with anomalies highlighted
        if 'market_apy' in data.columns:
            ax.plot(data.index, data['market_apy'], color=MUTED_BLUES[0], alpha=0.7, label='Normal')
            anomaly_dates = data.index[anomaly_labels == -1]
            anomaly_values = data.loc[anomaly_dates, 'market_apy']
            ax.scatter(anomaly_dates, anomaly_values, color='red', s=50, 
                      alpha=0.8, label='Anomalies', zorder=5)
        
        ax.set_ylabel('APY (%)')
        ax.set_title('Anomaly Detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        format_date_axis(ax)
    
    def _plot_correlation_heatmap(self, ax):
        """Plot correlation heatmap of key features."""
        # Select key APY columns for correlation
        apy_cols = [col for col in self.features_df.columns if '_apy' in col and 'ma_' not in col][:8]
        corr_data = self.features_df[apy_cols].corr()
        
        im = ax.imshow(corr_data.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation')
        
        # Set ticks and labels
        ax.set_xticks(range(len(corr_data.columns)))
        ax.set_yticks(range(len(corr_data.columns)))
        ax.set_xticklabels([col.replace('_apy', '') for col in corr_data.columns], 
                          rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([col.replace('_apy', '') for col in corr_data.columns], 
                          fontsize=8)
        
        # Add correlation values
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                ax.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                       ha='center', va='center', fontsize=7)
        
        ax.set_title('Pool APY Correlation Matrix')
    
    def _plot_time_series_analysis(self, ax):
        """Plot time series analysis with trends."""
        if 'market_apy' in self.features_df.columns:
            # Plot market APY with moving averages
            data = self.features_df[['market_apy', 'market_apy_ma_30d']].dropna()
            
            ax.plot(data.index, data['market_apy'], color=MUTED_BLUES[0], 
                   alpha=0.7, linewidth=1, label='Market APY')
            ax.plot(data.index, data['market_apy_ma_30d'], color=MUTED_BLUES[2], 
                   linewidth=2, label='30-day MA')
            
            # Add volatility bands
            if 'market_volatility' in self.features_df.columns:
                # Align volatility data with the main data
                vol_data = self.features_df.loc[data.index, 'market_volatility'].dropna()
                aligned_data = data.loc[vol_data.index]
                
                if not aligned_data.empty and not vol_data.empty:
                    upper_band = aligned_data['market_apy_ma_30d'] + vol_data
                    lower_band = aligned_data['market_apy_ma_30d'] - vol_data
                    
                    ax.fill_between(aligned_data.index, upper_band, lower_band, 
                                   alpha=0.2, color=MUTED_BLUES[1], label='Volatility Bands')
        
        ax.set_ylabel('APY (%)')
        ax.set_title('Market APY Time Series Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        format_date_axis(ax)
    
    def run_complete_analysis(self) -> None:
        """
        Run the complete ML analysis pipeline.
        """
        print("=" * 60)
        print("DeFi Machine Learning Analysis Pipeline")
        print("=" * 60)
        
        # Load and prepare data
        if not self.load_and_prepare_data():
            return
        
        print_data_summary(self.features_df, "ML Features")
        
        # Perform analyses
        clustering_results = self.perform_clustering_analysis()
        prediction_results = self.build_apy_prediction_model()
        anomaly_results = self.detect_anomalies()
        
        # Create comprehensive plots
        self.create_comprehensive_plots(clustering_results, prediction_results, anomaly_results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        if clustering_results:
            print(f"✓ Clustering Analysis: {clustering_results['silhouette_score']:.3f} silhouette score")
        
        if prediction_results:
            best_model = max([name for name in prediction_results.keys() if isinstance(prediction_results[name], dict)],
                           key=lambda x: prediction_results[x]['test_r2'])
            print(f"✓ Best Prediction Model: {best_model} (R² = {prediction_results[best_model]['test_r2']:.3f})")
        
        if anomaly_results:
            n_anomalies = np.sum(anomaly_results['isolation_forest_labels'] == -1)
            print(f"✓ Anomaly Detection: {n_anomalies} anomalies detected")
        
        print("✓ Comprehensive visualizations created")
        print("\nAnalysis complete!")


def main():
    """Main function to run the DeFi ML analysis."""
    analyzer = DeFiMLAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()