import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Tuple

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess ad clickstream data"""
    df_clean = df.copy()
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    df_clean[categorical_cols] = df_clean[categorical_cols].fillna('unknown')
    
    # Remove outliers using IQR method
    for col in ['age', 'income', 'website_visits']:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    
    print(f"Preprocessing complete. Shape: {df_clean.shape}")
    return df_clean

def create_treatment_control_split(df: pd.DataFrame, 
                                 treatment_col: str = 'treatment',
                                 outcome_col: str = 'conversion') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into treatment and control groups"""
    treatment_group = df[df[treatment_col] == 1].copy()
    control_group = df[df[treatment_col] == 0].copy()
    
    print(f"Treatment group size: {len(treatment_group)}")
    print(f"Control group size: {len(control_group)}")
    print(f"Treatment conversion rate: {treatment_group[outcome_col].mean():.3f}")
    print(f"Control conversion rate: {control_group[outcome_col].mean():.3f}")
    print(f"Naive uplift: {(treatment_group[outcome_col].mean() - control_group[outcome_col].mean()):.3f}")
    
    return treatment_group, control_group

def balance_check(df: pd.DataFrame, treatment_col: str = 'treatment') -> pd.DataFrame:
    """Check balance of covariates between treatment and control groups"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != treatment_col]
    
    balance_stats = []
    for col in numeric_cols:
        treated_mean = df[df[treatment_col] == 1][col].mean()
        control_mean = df[df[treatment_col] == 0][col].mean()
        
        treated_std = df[df[treatment_col] == 1][col].std()
        control_std = df[df[treatment_col] == 0][col].std()
        
        # Standardized mean difference
        pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
        smd = (treated_mean - control_mean) / pooled_std
        
        balance_stats.append({
            'variable': col,
            'treated_mean': treated_mean,
            'control_mean': control_mean,
            'standardized_mean_diff': smd,
            'imbalanced': abs(smd) > 0.1  # Common threshold
        })
    
    return pd.DataFrame(balance_stats)
