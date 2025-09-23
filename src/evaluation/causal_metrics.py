import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List

def compute_ate(treated_outcomes: np.ndarray, 
               control_outcomes: np.ndarray) -> Dict:
    """Compute Average Treatment Effect with confidence intervals"""
    
    ate = np.mean(treated_outcomes) - np.mean(control_outcomes)
    
    # Standard error calculation
    n_treated = len(treated_outcomes)
    n_control = len(control_outcomes)
    
    pooled_var = ((n_treated - 1) * np.var(treated_outcomes, ddof=1) + 
                  (n_control - 1) * np.var(control_outcomes, ddof=1)) / (n_treated + n_control - 2)
    
    ate_se = np.sqrt(pooled_var * (1/n_treated + 1/n_control))
    
    # Confidence intervals
    df = n_treated + n_control - 2
    t_critical = stats.t.ppf(0.975, df)
    
    ate_ci_lower = ate - t_critical * ate_se
    ate_ci_upper = ate + t_critical * ate_se
    
    # Statistical significance
    t_stat = ate / ate_se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    return {
        'ate': ate,
        'ate_se': ate_se,
        'ate_ci_lower': ate_ci_lower,
        'ate_ci_upper': ate_ci_upper,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def lift_calculation(treated_outcomes: np.ndarray,
                    control_outcomes: np.ndarray) -> Dict:
    """Calculate percentage lift and incremental metrics"""
    
    control_rate = np.mean(control_outcomes)
    treated_rate = np.mean(treated_outcomes)
    
    absolute_lift = treated_rate - control_rate
    relative_lift = (absolute_lift / control_rate) * 100 if control_rate > 0 else 0
    
    # Incremental conversions per 1000 impressions
    incremental_per_1000 = absolute_lift * 1000
    
    return {
        'control_rate': control_rate,
        'treated_rate': treated_rate,
        'absolute_lift': absolute_lift,
        'relative_lift_pct': relative_lift,
        'incremental_per_1000': incremental_per_1000
    }

def evaluate_model_performance(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             treatment_effects: np.ndarray = None) -> Dict:
    """Evaluate causal model performance"""
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Standard regression metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'r2_score': r2,
        'rmse': np.sqrt(mse)
    }
    
    # Treatment effect specific metrics
    if treatment_effects is not None:
        # Precision in Estimation of Heterogeneous Effect (PEHE)
        if len(treatment_effects) == len(y_true):
            pehe = np.mean((treatment_effects - (y_true - y_pred)) ** 2)
            metrics['pehe'] = pehe
            
        # Effect size distribution
        metrics['effect_mean'] = np.mean(treatment_effects)
        metrics['effect_std'] = np.std(treatment_effects)
        metrics['effect_range'] = np.ptp(treatment_effects)
    
    return metrics

def policy_value_evaluation(treatment_effects: np.ndarray,
                          costs: np.ndarray,
                          benefits: np.ndarray,
                          threshold: float = 0.0) -> Dict:
    """Evaluate policy value for treatment assignment"""
    
    # Policy: treat if predicted effect > threshold
    policy_assignment = treatment_effects > threshold
    
    # Calculate policy value
    total_benefit = np.sum(benefits[policy_assignment])
    total_cost = np.sum(costs[policy_assignment])
    net_value = total_benefit - total_cost
    
    # Treatment rate under policy
    treatment_rate = np.mean(policy_assignment)
    
    # ROI calculation
    roi = (total_benefit / total_cost - 1) * 100 if total_cost > 0 else 0
    
    return {
        'policy_value': net_value,
        'total_benefit': total_benefit,
        'total_cost': total_cost,
        'treatment_rate': treatment_rate,
        'roi_percent': roi,
        'treated_count': np.sum(policy_assignment)
    }

def bias_correction_assessment(naive_estimate: float,
                             causal_estimates: Dict) -> Dict:
    """Assess bias correction from different causal methods"""
    
    methods = ['psm', 'doubly_robust', 'causal_forest']
    bias_corrections = {}
    
    for method in methods:
        if method in causal_estimates:
            method_estimate = causal_estimates[method]['ate']
            bias_correction = naive_estimate - method_estimate
            bias_correction_pct = (bias_correction / naive_estimate) * 100
            
            bias_corrections[method] = {
                'estimate': method_estimate,
                'bias_correction': bias_correction,
                'bias_correction_pct': bias_correction_pct
            }
    
    return bias_corrections
