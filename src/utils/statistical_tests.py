import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List

def run_tests(df: pd.DataFrame, 
             treatment_col: str = 'treatment',
             outcome_col: str = 'conversion') -> Dict:
    """Run comprehensive statistical tests for causal analysis"""
    
    test_results = {}
    
    # 1. Balance tests
    test_results['balance_tests'] = run_balance_tests(df, treatment_col)
    
    # 2. Overlap tests
    test_results['overlap_tests'] = test_overlap_assumption(df, treatment_col)
    
    # 3. Parallel trends (if panel data)
    if 'time_period' in df.columns:
        test_results['parallel_trends'] = test_parallel_trends(df, treatment_col, outcome_col)
    
    # 4. Common support
    test_results['common_support'] = test_common_support(df, treatment_col)
    
    return test_results

def run_balance_tests(df: pd.DataFrame, treatment_col: str) -> Dict:
    """Test covariate balance between treatment groups"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != treatment_col]
    
    balance_results = {}
    
    for col in numeric_cols:
        treated = df[df[treatment_col] == 1][col]
        control = df[df[treatment_col] == 0][col]
        
        # T-test for mean differences
        t_stat, t_pvalue = stats.ttest_ind(treated, control)
        
        # Kolmogorov-Smirnov test for distribution differences
        ks_stat, ks_pvalue = stats.ks_2samp(treated, control)
        
        # Standardized mean difference
        pooled_std = np.sqrt((treated.var() + control.var()) / 2)
        smd = (treated.mean() - control.mean()) / pooled_std
        
        balance_results[col] = {
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'standardized_mean_diff': smd,
            'balanced': (abs(smd) < 0.1) and (t_pvalue > 0.05)
        }
    
    # Overall balance assessment
    imbalanced_vars = sum([1 for result in balance_results.values() if not result['balanced']])
    overall_balanced = imbalanced_vars / len(balance_results) < 0.2  # Less than 20% imbalanced
    
    balance_results['overall_assessment'] = {
        'imbalanced_variables': imbalanced_vars,
        'total_variables': len(balance_results) - 1,  # Exclude overall_assessment itself
        'overall_balanced': overall_balanced
    }
    
    return balance_results

def test_overlap_assumption(df: pd.DataFrame, treatment_col: str) -> Dict:
    """Test overlap/positivity assumption"""
    
    if 'propensity_score' not in df.columns:
        return {'error': 'Propensity scores not available'}
    
    treated_ps = df[df[treatment_col] == 1]['propensity_score']
    control_ps = df[df[treatment_col] == 0]['propensity_score']
    
    # Calculate overlap region
    treated_min, treated_max = treated_ps.min(), treated_ps.max()
    control_min, control_max = control_ps.min(), control_ps.max()
    
    overlap_min = max(treated_min, control_min)
    overlap_max = min(treated_max, control_max)
    overlap_width = max(0, overlap_max - overlap_min)
    
    # Proportion in common support
    common_support_treated = ((treated_ps >= overlap_min) & (treated_ps <= overlap_max)).mean()
    common_support_control = ((control_ps >= overlap_min) & (control_ps <= overlap_max)).mean()
    
    # Extreme propensity scores
    extreme_low = (df['propensity_score'] < 0.1).sum()
    extreme_high = (df['propensity_score'] > 0.9).sum()
    
    return {
        'overlap_width': overlap_width,
        'common_support_treated': common_support_treated,
        'common_support_control': common_support_control,
        'extreme_low_count': extreme_low,
        'extreme_high_count': extreme_high,
        'overlap_adequate': overlap_width > 0.6 and common_support_treated > 0.8 and common_support_control > 0.8
    }

def test_parallel_trends(df: pd.DataFrame, 
                        treatment_col: str,
                        outcome_col: str,
                        time_col: str = 'time_period') -> Dict:
    """Test parallel trends assumption for difference-in-differences"""
    
    # Pre-treatment periods
    pre_treatment = df[df[time_col] < df[time_col].median()]
    
    if len(pre_treatment) < 10:
        return {'error': 'Insufficient pre-treatment data'}
    
    # Calculate trends for each group
    treated_trend = []
    control_trend = []
    time_points = sorted(pre_treatment[time_col].unique())
    
    for t in time_points:
        period_data = pre_treatment[pre_treatment[time_col] == t]
        treated_mean = period_data[period_data[treatment_col] == 1][outcome_col].mean()
        control_mean = period_data[period_data[treatment_col] == 0][outcome_col].mean()
        
        treated_trend.append(treated_mean)
        control_trend.append(control_mean)
    
    # Test for parallel trends
    if len(treated_trend) > 2:
        treated_diff = np.diff(treated_trend)
        control_diff = np.diff(control_trend)
        
        # Test if differences in trends are statistically significant
        if len(treated_diff) > 1:
            trend_test_stat, trend_pvalue = stats.ttest_rel(treated_diff, control_diff)
        else:
            trend_test_stat = treated_diff[0] - control_diff[0]
            trend_pvalue = None
        
        return {
            'treated_trend': treated_trend,
            'control_trend': control_trend,
            'trend_test_statistic': trend_test_stat,
            'trend_test_pvalue': trend_pvalue,
            'parallel_trends_ok': trend_pvalue is None or trend_pvalue > 0.05
        }
    
    return {'error': 'Insufficient time periods for trend analysis'}

def test_common_support(df: pd.DataFrame, treatment_col: str) -> Dict:
    """Test common support assumption"""
    
    if 'propensity_score' not in df.columns:
        return {'error': 'Propensity scores not available'}
    
    treated_ps = df[df[treatment_col] == 1]['propensity_score']
    control_ps = df[df[treatment_col] == 0]['propensity_score']
    
    # Trimming bounds (common approach: 0.1 to 0.9)
    lower_bound = 0.1
    upper_bound = 0.9
    
    # Units on common support
    on_support = ((df['propensity_score'] >= lower_bound) & 
                  (df['propensity_score'] <= upper_bound))
    
    support_stats = {
        'total_units': len(df),
        'units_on_support': on_support.sum(),
        'proportion_on_support': on_support.mean(),
        'treated_on_support': ((df[treatment_col] == 1) & on_support).sum(),
        'control_on_support': ((df[treatment_col] == 0) & on_support).sum(),
        'adequate_support': on_support.mean() > 0.8
    }
    
    return support_stats

def normality_tests(df: pd.DataFrame, 
                   variables: List[str]) -> Dict:
    """Test normality assumptions for key variables"""
    
    normality_results = {}
    
    for var in variables:
        if var in df.columns:
            data = df[var].dropna()
            
            # Shapiro-Wilk test (for smaller samples)
            if len(data) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(data)
            else:
                shapiro_stat, shapiro_p = None, None
            
            # Anderson-Darling test
            anderson_result = stats.anderson(data, dist='norm')
            
            # Jarque-Bera test
            jb_stat, jb_p = stats.jarque_bera(data)
            
            normality_results[var] = {
                'shapiro_statistic': shapiro_stat,
                'shapiro_pvalue': shapiro_p,
                'anderson_statistic': anderson_result.statistic,
                'anderson_critical_values': anderson_result.critical_values,
                'jarque_bera_statistic': jb_stat,
                'jarque_bera_pvalue': jb_p,
                'appears_normal': jb_p > 0.05 if jb_p else False
            }
    
    return normality_results

def heteroskedasticity_tests(residuals: np.ndarray, 
                           fitted_values: np.ndarray) -> Dict:
    """Test for heteroskedasticity in residuals"""
    
    # Breusch-Pagan test approximation
    # Regress squared residuals on fitted values
    squared_residuals = residuals ** 2
    
    # Simple correlation test
    correlation, correlation_p = stats.pearsonr(fitted_values, squared_residuals)
    
    # White test approximation (using squared fitted values)
    squared_fitted = fitted_values ** 2
    white_corr, white_p = stats.pearsonr(squared_fitted, squared_residuals)
    
    return {
        'bp_correlation': correlation,
        'bp_pvalue': correlation_p,
        'white_correlation': white_corr,
        'white_pvalue': white_p,
        'homoskedastic': (correlation_p > 0.05) and (white_p > 0.05)
    }
