import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

def load_data(path: str, data_type: str = 'csv') -> pd.DataFrame:
    """Load benchmark ad-clickstream datasets"""
    if data_type == 'csv':
        df = pd.read_csv(path)
    elif data_type == 'parquet':
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    print(f"Loaded {len(df)} records from {path}")
    return df

def simulate_ad_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic ad clickstream data for testing"""
    np.random.seed(seed)
    
    # User demographics and behavior
    age = np.random.normal(35, 12, n_samples).clip(18, 70)
    income = np.random.lognormal(10.5, 0.5, n_samples).clip(20000, 200000)
    website_visits = np.random.poisson(5, n_samples)
    past_purchases = np.random.poisson(2, n_samples)
    
    # Treatment assignment (ad exposure) - not random, biased by demographics
    treatment_propensity = 0.3 + 0.2 * (age > 30) + 0.1 * (income > 50000) + 0.05 * website_visits
    treatment = np.random.binomial(1, treatment_propensity)
    
    # Outcome (conversion) - affected by both treatment and confounders
    conversion_propensity = (0.02 + 0.06 * treatment + 0.01 * (age > 30) + 
                           0.015 * (income > 50000) + 0.005 * website_visits + 
                           0.01 * past_purchases + np.random.normal(0, 0.01, n_samples))
    conversion = np.random.binomial(1, conversion_propensity.clip(0, 1))
    
    return pd.DataFrame({
        'user_id': range(n_samples),
        'age': age,
        'income': income,
        'website_visits': website_visits,
        'past_purchases': past_purchases,
        'treatment': treatment,
        'conversion': conversion,
        'true_propensity': treatment_propensity,
        'true_effect': 0.06  # True causal effect
    })

def load_benchmark_dataset(dataset_name: str = 'criteo') -> pd.DataFrame:
    """Load popular benchmark datasets for causal inference in ads"""
    # Placeholder for real benchmark datasets
    if dataset_name == 'criteo':
        return simulate_ad_data(50000, seed=123)
    elif dataset_name == 'ipinyou':
        return simulate_ad_data(75000, seed=456)
    else:
        return simulate_ad_data(10000, seed=42)
