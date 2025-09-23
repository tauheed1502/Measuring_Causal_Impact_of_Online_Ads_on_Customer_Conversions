import numpy as np
import pandas as pd
from typing import Dict, List
from scipy import stats

def robustness_checks(df: pd.DataFrame,
                     confounders: List[str],
                     treatment_col: str = 'treatment',
                     outcome_col: str = 'conversion') -> Dict:
    """Comprehensive robustness checks for causal estimates"""
    
    results = {}
    
    # 1. Subset validation
    results['subset_validation'] = subset_robustness(df, confounders, treatment_col, outcome_col)
    
    # 2. Confounder sensitivity
    results['confounder_sensitivity'] = confounder_robustness(df, confounders, treatment_col, outcome_col)
    
    # 3. Sample splitting validation
    results['sample_splitting'] = sample_splitting_validation(df, confounders, treatment_col, outcome_col)
    
    # 4. Bandwidth sensitivity (for matching methods)
    results['bandwidth_sensitivity'] = bandwidth_sensitivity_test(df, confounders, treatment_col, outcome_col)
    
    return results

def subset_robustness(df: pd.DataFrame,
                     confounders: List[str],
                     treatment_col: str,
                     outcome_col: str) -> Dict:
    """Test robustness across different data subsets"""
    
    from src.models.doubly_robust import doubly_robust_estimation
    
    # Different subset criteria
    subsets = {
        'high_propensity': df['propensity_score'] > 0.3 if 'propensity_score' in df.columns else df.sample(frac=0.8),
        'balanced_sample': df.sample(frac=0.7, random_state=42),
        'recent_period': df.tail(int(len(df) * 0.6))  # Last 60% of data
    }
    
    subset_results = {}
    
    for subset_name, subset_df in subsets.items():
        if len(subset_df) > 100:  # Minimum sample size
            try:
                dr_results = doubly_robust_estimation(subset_df, confounders, treatment_col, outcome_col)
                subset_results[subset_name] = {
                    'ate': dr_results['ate'],
                    'ate_se': dr_results['ate_se'],
                    'sample_size': len(subset_df)
                }
            except Exception as e:
                subset_results[subset_name] = {'error': str(e)}
    
    return subset_results

def confounder_robustness(df: pd.DataFrame,
                         confounders: List[str],
                         treatment_col: str,
                         outcome_col: str) -> Dict:
    """Test sensitivity to confounder selection"""
    
    from src.models.doubly_robust import doubly_robust_estimation
    
    confounder_tests = {}
    
    # Test with reduced confounder sets
    for drop_count in [1, 2, 3]:
        if len(confounders) > drop_count:
            # Randomly drop confounders
            for trial in range(3):  # Multiple trials
                np.random.seed(42 + trial)
                reduced_confounders = np.random.choice(confounders, 
                                                     size=len(confounders) - drop_count, 
                                                     replace=False).tolist()
                
                try:
                    dr_results = doubly_robust_estimation(df, reduced_confounders, treatment_col, outcome_col)
                    test_name = f"drop_{drop_count}_trial_{trial}"
                    confounder_tests[test_name] = {
                        'ate': dr_results['ate'],
                        'confounders_used': len(reduced_confounders),
                        'dropped_vars': [c for c in confounders if c not in reduced_confounders]
                    }
                except Exception as e:
                    pass
    
    return confounder_tests

def sample_splitting_validation(df: pd.DataFrame,
                              confounders: List[str],
                              treatment_col: str,
                              outcome_col: str) -> Dict:
    """Cross-validation style robustness check"""
    
    from src.models.doubly_robust import doubly_robust_estimation
    
    n_splits = 5
    split_size = len(df) // n_splits
    
    split_results = []
    
    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < n_splits - 1 else len(df)
        
        split_df = df.iloc[start_idx:end_idx]
        
        try:
            dr_results = doubly_robust_estimation(split_df, confounders, treatment_col, outcome_col)
            split_results.append({
                'split': i,
                'ate': dr_results['ate'],
                'ate_se': dr_results['ate_se'],
                'sample_size': len(split_df)
            })
        except Exception as e:
            pass
    
    if split_results:
        ate_estimates = [r['ate'] for r in split_results]
        return {
            'split_estimates': split_results,
            'mean_ate': np.mean(ate_estimates),
            'std_ate': np.std(ate_estimates),
            'consistency_check': np.std(ate_estimates) < 0.02  # Low variance = consistent
        }
    
    return {'error': 'No successful splits'}

def bandwidth_sensitivity_test(df: pd.DataFrame,
                             confounders: List[str],
                             treatment_col: str,
                             outcome_col: str) -> Dict:
    """Test sensitivity to matching bandwidth/caliper"""
    
    from src.models.propensity_score import propensity_score_matching
    
    calipers = [0.05, 0.1, 0.15, 0.2, 0.25]
    bandwidth_results = {}
    
    for caliper in calipers:
        try:
            matched_data, psm_results = propensity_score_matching(
                df, confounders, treatment_col, outcome_col, caliper=caliper
            )
            
            bandwidth_results[f'caliper_{caliper}'] = {
                'ate': psm_results['ate'],
                'matched_pairs': psm_results['matched_pairs'],
                'matching_rate': psm_results['matching_rate']
            }
        except Exception as e:
            bandwidth_results[f'caliper_{caliper}'] = {'error': str(e)}
    
    return bandwidth_results
