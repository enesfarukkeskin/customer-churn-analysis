import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    A class to handle model training and evaluation.
    """
    
    def __init__(self, data: pd.DataFrame, target_column: str = 'churn',
                 test_size: float = 0.2, random_state: int = 42):
        """
        Initialize ModelTrainer.
        
        Args:
            data (pd.DataFrame): Input data for training
            target_column (str): Name of target column
            test_size (float): Proportion of dataset to include in the test split
            random_state (int): Random state for reproducibility
        """
        self.data = data
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_data(self) -> None:
        """
        Prepare data for training by splitting features and target.
        """
        logger.info("Preparing data for training")
        
        self.y = self.data[self.target_column]
        self.X = self.data.drop(self.target_column, axis=1)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, 
            random_state=self.random_state, stratify=self.y
        )
        
        # Scale the features
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns
        )
        
    def handle_imbalance(self, method: str = 'smote') -> None:
        """
        Handle class imbalance in the training data.
        
        Args:
            method (str): Method to handle imbalance ('smote' or 'class_weight')
        """
        if method == 'smote':
            logger.info("Applying SMOTE to handle class imbalance")
            smote = SMOTE(random_state=self.random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            
    def train_random_forest(self, params: Optional[Dict] = None) -> None:
        """
        Train a Random Forest model.
        
        Args:
            params (Dict, optional): Parameters for Random Forest model
        """
        logger.info("Training Random Forest model")
        
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state
            }
            
        self.model = RandomForestClassifier(**params)
        self.model.fit(self.X_train, self.y_train)
        
    def train_xgboost(self, params: Optional[Dict] = None) -> None:
        """
        Train an XGBoost model.
        
        Args:
            params (Dict, optional): Parameters for XGBoost model
        """
        logger.info("Training XGBoost model")
        
        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'random_state': self.random_state
            }
            
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        evaluation = {
            'classification_report': classification_report(self.y_test, y_pred),
            'roc_auc_score': roc_auc_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train, 
            cv=5, scoring='roc_auc'
        )
        evaluation['cv_scores_mean'] = cv_scores.mean()
        evaluation['cv_scores_std'] = cv_scores.std()
        
        return evaluation
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            pd.DataFrame: DataFrame containing feature importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.model.feature_importances_
            })
            return importance.sort_values('importance', ascending=False)
        else:
            logger.warning("Model doesn't have feature importance attribute")
            return pd.DataFrame()
        
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        try:
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    @staticmethod
    def load_model(filepath: str) -> Any:
        """
        Load a saved model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Any: Loaded model
        """
        try:
            model = joblib.load(filepath)
            logger.info(f"Model loaded successfully from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def make_predictions(self, new_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            new_data (pd.DataFrame): New data to make predictions on
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        # Scale the new data
        scaled_data = pd.DataFrame(
            self.scaler.transform(new_data),
            columns=new_data.columns
        )
        
        # Make predictions
        predictions = self.model.predict_proba(scaled_data)[:, 1]
        return predictions

if __name__ == "__main__":
  
    data = pd.read_csv("data/processed/churn_train_processed.csv")
    
    # Initialize and train model
    trainer = ModelTrainer(data)
    trainer.prepare_data()
    trainer.handle_imbalance()
    
    # Train and evaluate Random Forest
    trainer.train_random_forest()
    rf_eval = trainer.evaluate_model()
    print("Random Forest Performance:", rf_eval)
    
    # Save the model
    trainer.save_model("models/saved_models/random_forest.joblib")
    
    # Train and evaluate XGBoost
    trainer.train_xgboost()
    xgb_eval = trainer.evaluate_model()
    print("XGBoost Performance:", xgb_eval)
    
    # Save the model
    trainer.save_model("models/saved_models/xgboost.joblib")