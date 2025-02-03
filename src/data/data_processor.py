import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    A class to handle data loading and preprocessing operations.
    """
    
    def __init__(self, input_path: str, output_path: Optional[str] = None):
        """
        Initialize DataProcessor with input and output paths.
        
        Args:
            input_path (str): Path to raw data file
            output_path (Optional[str]): Path to save processed data
        """
        self.input_path = input_path
        self.output_path = output_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from the specified input path.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            logger.info(f"Loading data from {self.input_path}")
            self.data = pd.read_csv(self.input_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Returns:
            pd.DataFrame: Data with handled missing values
        """
        logger.info("Handling missing values")
        
        # Get numeric and categorical columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Fill missing values
        for col in numeric_cols:
            self.data[col] = self.data[col].fillna(0)
            
        logger.info("Missing values handled successfully")
        return self.data
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Returns:
            pd.DataFrame: Data without duplicates
        """
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates()
        removed_rows = initial_rows - len(self.data)
        
        logger.info(f"Removed {removed_rows} duplicate rows")
        return self.data
    
    def handle_outliers(self, columns: list, method: str = 'iqr') -> pd.DataFrame:
        """
        Handle outliers in specified columns.
        
        Args:
            columns (list): List of columns to check for outliers
            method (str): Method to handle outliers ('iqr' or 'zscore')
            
        Returns:
            pd.DataFrame: Data with handled outliers
        """
        logger.info(f"Handling outliers using {method} method")
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                self.data[col] = self.data[col].mask(z_scores > 3, self.data[col].mean())
                
        return self.data
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.
        
        Returns:
            Dict[str, Any]: Dictionary containing data information
        """
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object']).columns)
        }
        return info
    
    def save_processed_data(self) -> None:
        """
        Save processed data to specified output path.
        """
        if self.output_path:
            try:
                self.data.to_csv(self.output_path, index=False)
                logger.info(f"Data saved successfully to {self.output_path}")
            except Exception as e:
                logger.error(f"Error saving data: {str(e)}")
                raise
    
    def process(self) -> pd.DataFrame:
        """
        Run the complete data processing pipeline.
        
        Returns:
            pd.DataFrame: Processed data
        """
        logger.info("Starting data processing pipeline")
        
        self.load_data()
        self.remove_duplicates()
        self.handle_missing_values()
        
        # Handle outliers for numeric columns related to usage and spending
        outlier_columns = [
            'user_spendings', 
            'calls_outgoing_count',
            'calls_outgoing_duration',
            'sms_outgoing_count',
            'gprs_usage'
        ]
        self.handle_outliers(outlier_columns)
        
        if self.output_path:
            self.save_processed_data()
            
        logger.info("Data processing completed successfully")
        return self.data

if __name__ == "__main__":
    processor = DataProcessor(
        input_path="data/raw/churn_train.csv",
        output_path="data/processed/churn_train_processed.csv"
    )
    processed_data = processor.process()
    print(processor.get_data_info())