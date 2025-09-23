import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_predict
from typing import Dict

class DoublyRobustEstimator:
    def __init__(self, 
                 outcome_model='random_forest',
                 propensity_model='logistic'):
        
        # Outcome model
        if outcome_model == 'random_forest':
            self.outcome_model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif outcome_model == 'linear':
            self.outcome_model = LinearRegression()
        else:
            raise ValueError(f"Unsupported outcome model: {outcome_model}")
            
        # Propensity model
        if propensity_model == 'logistic':
            self.propensity_model = LogisticRegression(random_state=42, max_iter=1000)
        elif propensity_model == 'random_forest':
            self.propensity_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported propensity model: {propensity_model}")

def doubly_robust_estimation(df: pd.DataFrame,
                           confounders: list,
                           treatment_col: str = 'treatment',
                           outcome_col: str = 'conversion') -> Dict:
    """Implement doubly robust estimation"""
    
    X = df[confounders]
    T = df[treatment_col]
    Y = df[outcome_col]
    
    estimator = DoublyRobustEstimator()
    
    # Step 1: Estimate propensity scores using cross-validation
    propensity_scores = cross_val_predict(estimator.propensity_model, X, T, 
                                        cv=5, method='predict_proba')[:, 1]
    
    # Step 2: Estimate outcome models for both treatment groups
    # Treated group outcome model
    treated_indices = T == 1
    control_indices = T == 0
    
    # Fit outcome model on control group, predict for all
    estimator.outcome_model.fit(X[control_indices], Y[control_indices])
    mu0 = estimator.outcome_model.predict(X)  # E[Y|X,T=0]
    
    # Fit outcome model on treated group, predict for all
    estimator.outcome_model.fit(X[treated_indices], Y[treated_indices])
    mu1 = estimator.outcome_model.predict(X)  # E[Y|X,T=1]
    
    # Step 3: Calculate doubly robust estimator
    n = len(df)
    
    # ATE calculation using doubly robust formula
    # ATE = E[mu1(X) - mu0(X)] + E[(T/e(X)) * (Y - mu1(X))] - E[((1-T)/(1-e(X))) * (Y - mu0(X))]
    
    # Avoid division by zero
    propensity_scores = np.clip(propensity_scores, 0.01, 0.99)
    
    # Calculate components
    direct_effect = np.mean(mu1 - mu0)
    
    treated_residual = np.mean((T / propensity_scores) * (Y - mu1))
    control_residual = np.mean(((1 - T) / (1 - propensity_scores)) * (Y - mu0))
    
    ate = direct_effect + treated_residual - control_residual
    
    # Calculate individual treatment effects for variance estimation
    individual_effects = (mu1 - mu0 + 
                         (T / propensity_scores) * (Y - mu1) - 
                         ((1 - T) / (1 - propensity_scores)) * (Y - mu0))
    
    ate_variance = np.var(individual_effects) / n
    ate_se = np.sqrt(ate_variance)
    
    results = {
        'ate': ate,
        'ate_se': ate_se,
        'ate_ci_lower': ate - 1.96 * ate_se,
        'ate_ci_upper': ate + 1.96 * ate_se,
        'direct_effect': direct_effect,
        'treated_residual': treated_residual,
        'control_residual': control_residual,
        'propensity_scores': propensity_scores,
        'mu0': mu0,
        'mu1': mu1
    }
    
    print(f"Doubly Robust Results - ATE: {ate:.4f} ± {ate_se:.4f}")
    print(f"95% CI: [{results['ate_ci_lower']:.4f}, {results['ate_ci_upper']:.4f}]")
    
    return results

def aipw_estimation(df: pd.DataFrame,
                   confounders: list,
                   treatment_col: str = 'treatment',
                   outcome_col: str = 'conversion') -> Dict:
    """Augmented Inverse Probability Weighting (AIPW) estimation"""
    
    # This is essentially the same as doubly robust estimation
    # AIPW is the formal name for the doubly robust estimator
    return doubly_robust_estimation(df, confounders, treatment_col, outcome_col)
