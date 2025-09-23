import numpy as np
import pandas as pd
from econml.grf import CausalForest
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import Dict, Optional

def fit_causal_forest(df: pd.DataFrame,
                     confounders: list,
                     treatment_col: str = 'treatment',
                     outcome_col: str = 'conversion',
                     model_type: str = 'standard') -> Dict:
    """Fit causal forest using econml"""
    
    X = df[confounders].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    
    if model_type == 'standard':
        # Standard Causal Forest
        est = CausalForest(
            n_estimators=100,
            criterion='mse',
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        # Fit the model
        est.fit(X, T, Y)
        
        # Predict treatment effects
        treatment_effects = est.effect(X)
        
    elif model_type == 'dml':
        # Double ML Causal Forest
        est = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, random_state=42),
            model_t=RandomForestClassifier(n_estimators=100, random_state=42),
            n_estimators=100,
            criterion='mse',
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        # Fit the model
        est.fit(Y, T, X=X)
        
        # Predict treatment effects
        treatment_effects = est.effect(X)
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Calculate aggregate treatment effect
    ate = np.mean(treatment_effects)
    ate_se = np.std(treatment_effects) / np.sqrt(len(treatment_effects))
    
    # Calculate heterogeneity statistics
    effect_variance = np.var(treatment_effects)
    heterogeneity_pvalue = calculate_heterogeneity_test(treatment_effects)
    
    # Confidence intervals
    ate_ci_lower = ate - 1.96 * ate_se
    ate_ci_upper = ate + 1.96 * ate_se
    
    results = {
        'ate': ate,
        'ate_se': ate_se,
        'ate_ci_lower': ate_ci_lower,
        'ate_ci_upper': ate_ci_upper,
        'treatment_effects': treatment_effects,
        'effect_variance': effect_variance,
        'heterogeneity_pvalue': heterogeneity_pvalue,
        'model': est
    }
    
    print(f"Causal Forest Results - ATE: {ate:.4f} ± {ate_se:.4f}")
    print(f"Effect heterogeneity p-value: {heterogeneity_pvalue:.4f}")
    
    return results

def calculate_heterogeneity_test(treatment_effects: np.ndarray) -> float:
    """Test for treatment effect heterogeneity"""
    from scipy import stats
    
    # Simple test: compare variance to what we'd expect under homogeneous effects
    # Under null hypothesis of no heterogeneity, effects should be similar
    mean_effect = np.mean(treatment_effects)
    
    # Chi-square test for heterogeneity
    chi_stat = np.sum((treatment_effects - mean_effect) ** 2) / np.var(treatment_effects)
    df = len(treatment_effects) - 1
    p_value = 1 - stats.chi2.cdf(chi_stat, df)
    
    return p_value

def analyze_heterogeneous_effects(df: pd.DataFrame,
                                treatment_effects: np.ndarray,
                                confounders: list,
                                n_groups: int = 4) -> Dict:
    """Analyze heterogeneous treatment effects by subgroups"""
    
    df_effects = df.copy()
    df_effects['treatment_effect'] = treatment_effects
    
    heterogeneity_analysis = {}
    
    # Analyze by quartiles of key variables
    key_vars = ['age', 'income', 'website_visits']
    for var in key_vars:
        if var in df_effects.columns:
            df_effects[f'{var}_quartile'] = pd.qcut(df_effects[var], q=n_groups, labels=False)
            group_effects = df_effects.groupby(f'{var}_quartile')['treatment_effect'].agg(['mean', 'std', 'count'])
            heterogeneity_analysis[var] = group_effects
    
    # Find most/least responsive subgroups
    top_quartile_threshold = np.percentile(treatment_effects, 75)
    bottom_quartile_threshold = np.percentile(treatment_effects, 25)
    
    high_responders = df_effects[df_effects['treatment_effect'] >= top_quartile_threshold]
    low_responders = df_effects[df_effects['treatment_effect'] <= bottom_quartile_threshold]
    
    heterogeneity_analysis['high_responders'] = high_responders[confounders].mean()
    heterogeneity_analysis['low_responders'] = low_responders[confounders].mean()
    
    return heterogeneity_analysis
