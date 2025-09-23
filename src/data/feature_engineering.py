import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from typing import List

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features for causal analysis"""
    df_features = df.copy()
    
    # Interaction features
    if 'age' in df_features.columns and 'income' in df_features.columns:
        df_features['age_income_interaction'] = df_features['age'] * df_features['income'] / 1000
    
    # Binning continuous variables
    if 'age' in df_features.columns:
        df_features['age_group'] = pd.cut(df_features['age'], 
                                        bins=[0, 25, 35, 50, 100], 
                                        labels=['young', 'medium', 'mature', 'senior'])
    
    if 'income' in df_features.columns:
        df_features['income_quartile'] = pd.qcut(df_features['income'], 
                                               q=4, labels=['low', 'medium', 'high', 'very_high'])
    
    # User engagement score
    if all(col in df_features.columns for col in ['website_visits', 'past_purchases']):
        scaler = StandardScaler()
        engagement_features = scaler.fit_transform(df_features[['website_visits', 'past_purchases']])
        df_features['engagement_score'] = engagement_features.sum(axis=1)
    
    # High-value customer indicator
    if 'income' in df_features.columns and 'past_purchases' in df_features.columns:
        df_features['high_value_customer'] = ((df_features['income'] > df_features['income'].quantile(0.75)) & 
                                            (df_features['past_purchases'] > 2)).astype(int)
    
    # One-hot encode categorical variables
    categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns
    df_features = pd.get_dummies(df_features, columns=categorical_cols, prefix=categorical_cols)
    
    print(f"Feature engineering complete. New shape: {df_features.shape}")
    return df_features

def select_confounders(df: pd.DataFrame, 
                      treatment_col: str = 'treatment',
                      outcome_col: str = 'conversion') -> List[str]:
    """Select relevant confounding variables"""
    # Exclude treatment, outcome, and ID columns
    exclude_cols = [treatment_col, outcome_col, 'user_id', 'true_propensity', 'true_effect']
    confounder_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove low-variance features
    numeric_confounders = df[confounder_cols].select_dtypes(include=[np.number])
    low_variance_cols = numeric_confounders.columns[numeric_confounders.var() < 0.01].tolist()
    confounder_cols = [col for col in confounder_cols if col not in low_variance_cols]
    
    print(f"Selected {len(confounder_cols)} confounding variables")
    return confounder_cols
