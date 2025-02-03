import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureBuilder:
    """
    A class to handle feature engineering operations.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize FeatureBuilder with input data.
        
        Args:
            data (pd.DataFrame): Input data for feature engineering
        """
        self.data = data.copy()
        self.original_columns = data.columns.tolist()
        self.created_features = []
        
    def create_ratio_features(self) -> None:
        """
        Create ratio-based features.
        """
        logger.info("Creating ratio features")
        
        # User activity ratios
        self.data['active_days_ratio'] = (self.data['user_lifetime'] - 
                                        self.data['user_no_outgoing_activity_in_days']) / (self.data['user_lifetime'] + 1)
        
        # Usage intensity ratios
        self.data['calls_per_day'] = self.data['calls_outgoing_count'] / (self.data['user_lifetime'] + 1)
        self.data['sms_per_day'] = self.data['sms_outgoing_count'] / (self.data['user_lifetime'] + 1)
        self.data['gprs_per_day'] = self.data['gprs_usage'] / (self.data['user_lifetime'] + 1)
        
        # Financial ratios
        self.data['spending_per_day'] = self.data['user_spendings'] / (self.data['user_lifetime'] + 1)
        self.data['reload_frequency'] = self.data['reloads_count'] / (self.data['user_lifetime'] + 1)
        self.data['avg_reload_amount'] = self.data['reloads_sum'] / (self.data['reloads_count'] + 1)
        
        self.created_features.extend([
            'active_days_ratio', 'calls_per_day', 'sms_per_day', 'gprs_per_day',
            'spending_per_day', 'reload_frequency', 'avg_reload_amount'
        ])
        
    def create_usage_pattern_features(self) -> None:
        """
        Create features related to usage patterns.
        """
        logger.info("Creating usage pattern features")
        
        # Communication preferences
        self.data['onnet_call_ratio'] = (self.data['calls_outgoing_to_onnet_count'] / 
                                        (self.data['calls_outgoing_count'] + 1))
        self.data['offnet_call_ratio'] = (self.data['calls_outgoing_to_offnet_count'] / 
                                         (self.data['calls_outgoing_count'] + 1))
        self.data['international_call_ratio'] = (self.data['calls_outgoing_to_abroad_count'] / 
                                               (self.data['calls_outgoing_count'] + 1))
        
        # Service usage diversity
        self.data['service_diversity'] = (
            (self.data['user_has_outgoing_calls'] > 0).astype(int) +
            (self.data['user_has_outgoing_sms'] > 0).astype(int) +
            (self.data['user_use_gprs'] > 0).astype(int)
        )
        
        self.created_features.extend([
            'onnet_call_ratio', 'offnet_call_ratio', 'international_call_ratio',
            'service_diversity'
        ])
        
    def create_customer_value_features(self) -> None:
        """
        Create features related to customer value and behavior.
        """
        logger.info("Creating customer value features")
        
        # Average spending per service
        self.data['avg_spending_per_call'] = (self.data['calls_outgoing_spendings'] / 
                                            (self.data['calls_outgoing_count'] + 1))
        self.data['avg_spending_per_sms'] = (self.data['sms_outgoing_spendings'] / 
                                           (self.data['sms_outgoing_count'] + 1))
        
        # Recent behavior features
        self.data['recent_calls_ratio'] = (self.data['last_100_calls_outgoing_duration'] / 
                                         (self.data['calls_outgoing_duration'] + 1))
        self.data['recent_sms_ratio'] = (self.data['last_100_sms_outgoing_count'] / 
                                       (self.data['sms_outgoing_count'] + 1))
        
        self.created_features.extend([
            'avg_spending_per_call', 'avg_spending_per_sms',
            'recent_calls_ratio', 'recent_sms_ratio'
        ])
        
    def create_interaction_features(self) -> None:
        """
        Create interaction features between existing features.
        """
        logger.info("Creating interaction features")
        
        # Spending and usage interactions
        self.data['spending_call_interaction'] = (self.data['user_spendings'] * 
                                                self.data['calls_per_day'])
        self.data['spending_service_interaction'] = (self.data['user_spendings'] * 
                                                   self.data['service_diversity'])
        
        self.created_features.extend([
            'spending_call_interaction', 'spending_service_interaction'
        ])
        
    def get_feature_info(self) -> Dict:
        """
        Get information about created features.
        
        Returns:
            Dict: Dictionary containing feature information
        """
        return {
            'original_features': self.original_columns,
            'created_features': self.created_features,
            'total_features': len(self.original_columns) + len(self.created_features)
        }
    
    def build_features(self, drop_original: bool = False) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            drop_original (bool): Whether to drop original features
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        logger.info("Starting feature engineering pipeline")
        
        self.create_ratio_features()
        self.create_usage_pattern_features()
        self.create_customer_value_features()
        self.create_interaction_features()
        
        if drop_original:
            self.data = self.data[self.created_features + ['churn']]
            
        logger.info("Feature engineering completed successfully")
        return self.data

if __name__ == "__main__":
  
    input_data = pd.read_csv("data/processed/churn_train_processed.csv")
    feature_builder = FeatureBuilder(input_data)
    engineered_data = feature_builder.build_features()
    print(feature_builder.get_feature_info())