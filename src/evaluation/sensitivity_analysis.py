import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats

def sensitivity_analysis(df: pd.DataFrame,
                        treatment_col: str = 'treatment',
                        outcome_col: str = 'conversion',
                        ate_estimate: float = None) -> Dict:
    """Rosenbaum bounds sensitivity analysis"""
    
    results = {}
    
    # 1. Rosenbaum bounds
    results['rosenbaum_bounds'] = rosenbaum_sensitivity_bounds(df, treatment_col, outcome_col)
    
    # 2. Omitted variable bias analysis
    results['omitted_variable_bias'] = omitted_variable_sensitivity(df, treatment_col, outcome_col, ate_estimate)
    
    # 3. Confounding strength simulation
    results['confounding_simulation'] = simulate_confounding_bias(df, treatment_col, outcome_col)
    
    return results

def rosenbaum_sensitivity_bounds(df: pd.DataFrame,
                               treatment_col: str,
                               outcome_col: str,
                               gamma_range: np.ndarray = None) -> Dict:
    """Calculate Rosenbaum bounds for hidden bias"""
    
    if gamma_range is None:
        gamma_range = np.arange(1.0, 3.1, 0.2)
    
    bounds = []
    
    # Calculate observed test statistic
    treated = df[df[treatment_col] == 1][outcome_col]
    control = df[df[treatment_col] == 0][outcome_col]
    
    observed_stat, observed_p = stats.mannwhitneyu(treated, control, alternative='two-sided')
    
    for gamma in gamma_range:
        # Calculate bounds under hidden bias of magnitude gamma
        # This is a simplified version - full implementation would use exact Rosenbaum formulas
        
        # Approximate bias adjustment
        n_treated = len(treated)
        n_control = len(control)
        
        # Bias factor from unmeasured confounding
        bias_factor = (gamma - 1) / (gamma + 1)
        
        # Adjusted p-value bounds
        p_upper = calculate_adjusted_pvalue(observed_stat, n_treated, n_control, bias_factor, direction='upper')
        p_lower = calculate_adjusted_pvalue(observed_stat, n_treated, n_control, bias_factor, direction='lower')
        
        bounds.append({
            'gamma': gamma,
            'p_value_upper': p_upper,
            'p_value_lower': p_lower,
            'significant_upper': p_upper < 0.05,
            'significant_lower': p_lower < 0.05
        })
    
    # Find critical gamma (where effect becomes non-significant)
    critical_gamma = None
    for bound in bounds:
        if not bound['significant_upper']:
            critical_gamma = bound['gamma']
            break
    
    return {
        'observed_p_value': observed_p,
        'bounds': bounds,
        'critical_gamma': critical_gamma,
        'robust_to_bias': critical_gamma is None or critical_gamma > 2.0
    }

def calculate_adjusted_pvalue(test_stat: float, 
                            n_treated: int, 
                            n_control: int,
                            bias_factor: float,
                            direction: str) -> float:
    """Calculate bias-adjusted p-value (simplified version)"""
    
    # This is a simplified approximation
    # Full implementation would use exact Rosenbaum formulas
    
    adjustment = bias_factor * np.sqrt(n_treated * n_control / (n_treated + n_control))
    
    if direction == 'upper':
        adjusted_stat = test_stat - adjustment
    else:
        adjusted_stat = test_stat + adjustment
    
    # Convert back to p-value (approximation)
    z_score = (adjusted_stat - n_treated * n_control / 2) / np.sqrt(n_treated * n_control * (n_treated + n_control + 1) / 12)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return np.clip(p_value, 0, 1)

def omitted_variable_sensitivity(df: pd.DataFrame,
                               treatment_col: str,
                               outcome_col: str,
                               ate_estimate: float) -> Dict:
    """Assess sensitivity to omitted variable bias"""
    
    # Simulate various strengths of omitted confounders
    bias_scenarios = []
    
    # Different correlation strengths between omitted variable and treatment/outcome
    correlations = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5)]
    
    for r_treat, r_outcome in correlations:
        # Generate omitted confounder
        n = len(df)
        omitted_var = np.random.normal(0, 1, n)
        
        # Induce correlation with treatment and outcome
        treatment_corr = df[treatment_col] * r_treat + omitted_var * np.sqrt(1 - r_treat**2)
        outcome_corr = df[outcome_col] * r_outcome + omitted_var * np.sqrt(1 - r_outcome**2)
        
        # Estimate bias
        # Simplified bias formula: bias ≈ corr(U,T) * corr(U,Y) * var(Y) / var(T)
        bias_estimate = r_treat * r_outcome * np.var(df[outcome_col]) / max(np.var(df[treatment_col]), 0.01)
        
        bias_scenarios.append({
            'r_treatment': r_treat,
            'r_outcome': r_outcome,
            'estimated_bias': bias_estimate,
            'adjusted_ate': ate_estimate - bias_estimate if ate_estimate else None
        })
    
    return {
        'original_ate': ate_estimate,
        'bias_scenarios': bias_scenarios,
        'max_bias': max([s['estimated_bias'] for s in bias_scenarios]),
        'robust_conclusion': all([abs(s['estimated_bias']) < 0.02 for s in bias_scenarios])
    }

def simulate_confounding_bias(df: pd.DataFrame,
                            treatment_col: str,
                            outcome_col: str,
                            n_simulations: int = 1000) -> Dict:
    """Monte Carlo simulation of confounding bias"""
    
    bias_estimates = []
    
    for _ in range(n_simulations):
        # Generate random confounding strength
        confounder_strength = np.random.uniform(0, 0.5)
        
        # Create synthetic confounder
        n = len(df)
        synthetic_confounder = np.random.beta(2, 5, n)  # Skewed distribution
        
        # Introduce bias based on confounder
        biased_treatment = df[treatment_col] + confounder_strength * synthetic_confounder
        biased_outcome = df[outcome_col] + confounder_strength * synthetic_confounder
        
        # Calculate biased effect
        treated_mean = biased_outcome[df[treatment_col] == 1].mean()
        control_mean = biased_outcome[df[treatment_col] == 0].mean()
        biased_effect = treated_mean - control_mean
        
        # Original effect
        original_treated = df[df[treatment_col] == 1][outcome_col].mean()
        original_control = df[df[treatment_col] == 0][outcome_col].mean()
        original_effect = original_treated - original_control
        
        bias = biased_effect - original_effect
        bias_estimates.append(bias)
    
    return {
        'mean_bias': np.mean(bias_estimates),
        'bias_std': np.std(bias_estimates),
        'bias_95_ci': [np.percentile(bias_estimates, 2.5), np.percentile(bias_estimates, 97.5)],
        'max_bias': np.max(np.abs(bias_estimates)),
        'bias_distribution': bias_estimates
    }

def placebo_tests(df: pd.DataFrame,
                 confounders: list,
                 treatment_col: str = 'treatment',
                 outcome_col: str = 'conversion') -> Dict:
    """Conduct placebo tests for causal identification"""
    
    from src.models.doubly_robust import doubly_robust_estimation
    
    placebo_results = {}
    
    # 1. Fake outcome placebo test
    fake_outcomes = ['age', 'income', 'website_visits']  # Pre-treatment variables
    
    for fake_outcome in fake_outcomes:
        if fake_outcome in df.columns and fake_outcome in confounders:
            # Remove the fake outcome from confounders for this test
            placebo_confounders = [c for c in confounders if c != fake_outcome]
            
            try:
                result = doubly_robust_estimation(df, placebo_confounders, treatment_col, fake_outcome)
                placebo_results[f'fake_outcome_{fake_outcome}'] = {
                    'ate': result['ate'],
                    'p_value_approx': 2 * (1 - stats.norm.cdf(abs(result['ate'] / result['ate_se']))),
                    'should_be_zero': abs(result['ate']) < 0.01
                }
            except Exception as e:
                placebo_results[f'fake_outcome_{fake_outcome}'] = {'error': str(e)}
    
    # 2. Random treatment assignment placebo
    df_placebo = df.copy()
    np.random.seed(42)
    df_placebo['random_treatment'] = np.random.binomial(1, df[treatment_col].mean(), len(df))
    
    try:
        random_result = doubly_robust_estimation(df_placebo, confounders, 'random_treatment', outcome_col)
        placebo_results['random_treatment'] = {
            'ate': random_result['ate'],
            'p_value_approx': 2 * (1 - stats.norm.cdf(abs(random_result['ate'] / random_result['ate_se']))),
            'should_be_zero': abs(random_result['ate']) < 0.01
        }
    except Exception as e:
        placebo_results['random_treatment'] = {'error': str(e)}
    
    return placebo_results
