import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Visualizer:
    """
    A class to handle data visualization.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize Visualizer with input data.
        
        Args:
            data (pd.DataFrame): Input data for visualization
        """
        self.data = data
        self.figure_size = (12, 6)
        plt.style.use('seaborn')
        
    def plot_feature_distributions(self, features: List[str], 
                                 target_col: str = 'churn') -> None:
        """
        Plot distribution of features by target variable.
        
        Args:
            features (List[str]): List of features to plot
            target_col (str): Name of target column
        """
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.ravel()
        
        for idx, feature in enumerate(features):
            sns.boxplot(x=target_col, y=feature, data=self.data, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {feature} by {target_col}')
            
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_matrix(self, features: Optional[List[str]] = None) -> None:
        """
        Plot correlation matrix for selected features.
        
        Args:
            features (List[str], optional): List of features to include
        """
        if features is None:
            correlation_matrix = self.data.corr()
        else:
            correlation_matrix = self.data[features].corr()
            
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                              top_n: int = 15) -> None:
        """
        Plot feature importance.
        
        Args:
            importance_df (pd.DataFrame): DataFrame with feature importance scores
            top_n (int): Number of top features to show
        """
        plt.figure(figsize=self.figure_size)
        
        importance_df = importance_df.head(top_n)
        sns.barplot(x='importance', y='feature', data=importance_df)
        
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, conf_matrix: np.ndarray) -> None:
        """
        Plot confusion matrix.
        
        Args:
            conf_matrix (np.ndarray): Confusion matrix array
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc: float) -> None:
        """
        Plot ROC curve.
        
        Args:
            fpr (np.ndarray): False positive rates
            tpr (np.ndarray): True positive rates
            auc (float): Area under curve score
        """
        plt.figure(figsize=self.figure_size)
        
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_numerical_feature_distributions(self, features: List[str]) -> None:
        """
        Plot distributions of numerical features.
        
        Args:
            features (List[str]): List of numerical features to plot
        """
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.ravel()
        
        for idx, feature in enumerate(features):
            sns.histplot(data=self.data, x=feature, ax=axes[idx], kde=True)
            axes[idx].set_title(f'Distribution of {feature}')
            
        plt.tight_layout()
        plt.show()
        
    def plot_time_series(self, date_column: str, value_column: str, 
                        agg_func: str = 'mean') -> None:
        """
        Plot time series data.
        
        Args:
            date_column (str): Name of date column
            value_column (str): Name of value column
            agg_func (str): Aggregation function to use
        """
        time_series = (self.data.groupby(date_column)[value_column]
                      .agg(agg_func).reset_index())
        
        plt.figure(figsize=self.figure_size)
        plt.plot(time_series[date_column], time_series[value_column], 
                marker='o', linestyle='-')
        
        plt.title(f'Time Series of {value_column} ({agg_func})')
        plt.xlabel(date_column)
        plt.ylabel(value_column)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_feature_relationships(self, feature_x: str, feature_y: str, 
                                 hue: Optional[str] = None) -> None:
        """
        Plot relationship between two features.
        
        Args:
            feature_x (str): Name of feature for x-axis
            feature_y (str): Name of feature for y-axis
            hue (str, optional): Name of feature for color coding
        """
        plt.figure(figsize=self.figure_size)
        
        sns.scatterplot(data=self.data, x=feature_x, y=feature_y, hue=hue)
        plt.title(f'Relationship between {feature_x} and {feature_y}')
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.tight_layout()
        plt.show()

    def plot_churn_rate_by_feature(self, feature: str, bins: int = 10) -> None:
        """
        Plot churn rate by feature bins.
        
        Args:
            feature (str): Feature to analyze
            bins (int): Number of bins for numerical features
        """
        plt.figure(figsize=self.figure_size)
        
        if self.data[feature].dtype in ['int64', 'float64']:
            # For numerical features
            self.data['bins'] = pd.qcut(self.data[feature], bins, duplicates='drop')
            churn_rate = self.data.groupby('bins')['churn'].mean()
            
            plt.bar(range(len(churn_rate)), churn_rate)
            plt.xticks(range(len(churn_rate)), 
                      [str(val) for val in churn_rate.index], 
                      rotation=45)
            self.data.drop('bins', axis=1, inplace=True)
        else:
            # For categorical features
            churn_rate = self.data.groupby(feature)['churn'].mean()
            plt.bar(range(len(churn_rate)), churn_rate)
            plt.xticks(range(len(churn_rate)), 
                      [str(val) for val in churn_rate.index], 
                      rotation=45)
            
        plt.title(f'Churn Rate by {feature}')
        plt.xlabel(feature)
        plt.ylabel('Churn Rate')
        plt.tight_layout()
        plt.show()

    def save_all_plots(self, output_dir: str, features: List[str]) -> None:
        """
        Save all plots to specified directory.
        
        Args:
            output_dir (str): Directory to save plots
            features (List[str]): List of features to analyze
        """
        logger.info(f"Saving plots to {output_dir}")
        
        # Save correlation matrix
        plt.figure()
        self.plot_correlation_matrix(features)
        plt.savefig(f"{output_dir}/correlation_matrix.png")
        plt.close()
        
        # Save feature distributions
        plt.figure()
        self.plot_numerical_feature_distributions(features)
        plt.savefig(f"{output_dir}/feature_distributions.png")
        plt.close()
        
        # Save churn rate plots for each feature
        for feature in features:
            plt.figure()
            self.plot_churn_rate_by_feature(feature)
            plt.savefig(f"{output_dir}/churn_rate_by_{feature}.png")
            plt.close()

if __name__ == "__main__":
  
    data = pd.read_csv("data/processed/churn_train_processed.csv")
    visualizer = Visualizer(data)
    
    # Example features for visualization
    important_features = [
        'user_lifetime',
        'user_spendings',
        'calls_outgoing_count',
        'sms_outgoing_count',
        'gprs_usage'
    ]
    
    # Create and save visualizations
    visualizer.plot_feature_distributions(important_features)
    visualizer.plot_correlation_matrix(important_features)
    visualizer.save_all_plots("visualizations", important_features)