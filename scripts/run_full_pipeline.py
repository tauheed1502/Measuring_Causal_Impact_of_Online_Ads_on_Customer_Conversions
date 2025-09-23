import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from typing import Dict
import yaml
import pickle

# Import our modules
from src.data.data_loader import load_data, simulate_ad_data
from src.data.preprocessing import preprocess, create_treatment_control_split
from src.data.feature_engineering import engineer_features, select_confounders
from src.models.propensity_score import propensity_score_matching
from src.models.doubly_robust import doubly_robust_estimation
from src.models.causal_forest import fit_causal_forest
from src.evaluation.causal_metrics import compute_ate, lift_calculation, bias_correction_assessment
from src.evaluation.robustness_tests import robustness_checks
from src.evaluation.sensitivity_analysis import sensitivity_analysis, placebo_tests
from src.utils.plotting import plot_results, plot_covariate_balance

def run_pipeline(config_path: str = 'config/experiment_config.yaml'):
    """Run the complete causal inference pipeline"""
    
    print("🚀 Starting Causal Ad Impact Analysis Pipeline")
    print("=" * 50)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Data Loading
    print("\n📊 Step 1: Loading Data")
    if config.get('use_simulated_data', True):
        df = simulate_ad_data(n_samples=config.get('sample_size', 10000))
        print(f"✅ Generated {len(df)} synthetic records")
    else:
        df = load_data(config['data_path'])
        print(f"✅ Loaded {len(df)} records from {config['data_path']}")
    
    # Step 2: Preprocessing
    print("\n🧹 Step 2: Data Preprocessing")
    df_clean = preprocess(df)
    
    # Step 3: Feature Engineering
    print("\n⚙️ Step 3: Feature Engineering")
    df_features = engineer_features(df_clean)
    confounders = select_confounders(df_features)
    print(f"✅ Selected {len(confounders)} confounding variables")
    
    # Step 4: Calculate Naive Estimate
    print("\n📈 Step 4: Naive Treatment Effect")
    treatment_group, control_group = create_treatment_control_split(df_features)
    naive_ate = treatment_group['conversion'].mean() - control_group['conversion'].mean()
    print(f"✅ Naive ATE: {naive_ate:.4f}")
    
    # Step 5: Causal Methods
    print("\n🎯 Step 5: Causal Inference Methods")
    results = {'naive': {'ate': naive_ate}}
    
    # Propensity Score Matching
    print("   Running Propensity Score Matching...")
    try:
        matched_data, psm_results = propensity_score_matching(df_features, confounders)
        results['psm'] = psm_results
        print(f"   ✅ PSM ATE: {psm_results['ate']:.4f}")
    except Exception as e:
        print(f"   ❌ PSM failed: {str(e)}")
    
    # Doubly Robust Estimation
    print("   Running Doubly Robust Estimation...")
    try:
        dr_results = doubly_robust_estimation(df_features, confounders)
        results['doubly_robust'] = dr_results
        print(f"   ✅ DR ATE: {dr_results['ate']:.4f} ± {dr_results['ate_se']:.4f}")
    except Exception as e:
        print(f"   ❌ DR failed: {str(e)}")
    
    # Causal Forest
    print("   Running Causal Forest...")
    try:
        cf_results = fit_causal_forest(df_features, confounders)
        results['causal_forest'] = cf_results
        print(f"   ✅ CF ATE: {cf_results['ate']:.4f} ± {cf_results['ate_se']:.4f}")
    except Exception as e:
        print(f"   ❌ CF failed: {str(e)}")
    
    # Step 6: Evaluation Metrics
    print("\n📊 Step 6: Evaluation Metrics")
    
    # Lift calculations
    lift_metrics = lift_calculation(treatment_group['conversion'], control_group['conversion'])
    print(f"✅ Relative Lift: {lift_metrics['relative_lift_pct']:.2f}%")
    
    # Bias correction assessment
    causal_estimates = {k: v for k, v in results.items() if k != 'naive'}
    bias_assessment = bias_correction_assessment(naive_ate, causal_estimates)
    print("✅ Bias correction completed")
    
    # Step 7: Robustness Checks
    print("\n🔍 Step 7: Robustness Checks")
    try:
        robustness_results = robustness_checks(df_features, confounders)
        results['robustness'] = robustness_results
        print("✅ Robustness checks completed")
    except Exception as e:
        print(f"❌ Robustness checks failed: {str(e)}")
    
    # Step 8: Sensitivity Analysis
    print("\n🎛️ Step 8: Sensitivity Analysis")
    try:
        sensitivity_results = sensitivity_analysis(df_features, ate_estimate=dr_results.get('ate'))
        results['sensitivity'] = sensitivity_results
        print("✅ Sensitivity analysis completed")
    except Exception as e:
        print(f"❌ Sensitivity analysis failed: {str(e)}")
    
    # Step 9: Placebo Tests
    print("\n🧪 Step 9: Placebo Tests")
    try:
        placebo_results = placebo_tests(df_features, confounders)
        results['placebo'] = placebo_results
        print("✅ Placebo tests completed")
    except Exception as e:
        print(f"❌ Placebo tests failed: {str(e)}")
    
    # Step 10: Generate Plots
    print("\n📈 Step 10: Generating Visualizations")
    try:
        # Add propensity scores to dataframe for plotting
        if 'doubly_robust' in results:
            df_features['propensity_score'] = results['doubly_robust']['propensity_scores']
        
        plot_results(results, save_path='results/figures/causal_analysis_results.png')
        plot_covariate_balance(df_features, confounders, save_path='results/figures/covariate_balance.png')
        print("✅ Visualizations saved")
    except Exception as e:
        print(f"❌ Plotting failed: {str(e)}")
    
    # Step 11: Save Results
    print("\n💾 Step 11: Saving Results")
    
    # Save detailed results
    with open('results/causal_analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary
    summary = create_summary_report(results, bias_assessment, lift_metrics)
    with open('results/analysis_summary.yaml', 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    print("✅ Results saved")
    
    # Final Summary
    print("\n" + "=" * 50)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"📊 Naive Estimate: {naive_ate:.4f}")
    if 'doubly_robust' in results:
        print(f"🎯 Causal Estimate (DR): {results['doubly_robust']['ate']:.4f}")
        bias_correction = naive_ate - results['doubly_robust']['ate']
        print(f"⚖️ Bias Correction: {bias_correction:.4f} ({bias_correction/naive_ate*100:.1f}%)")
    print(f"📈 Business Impact: {lift_metrics['incremental_per_1000']:.2f} incremental conversions per 1000 impressions")
    
    return results

def create_summary_report(results: Dict, bias_assessment: Dict, lift_metrics: Dict) -> Dict:
    """Create executive summary of results"""
    
    summary = {
        'executive_summary': {
            'naive_estimate': results['naive']['ate'],
            'causal_estimate': results.get('doubly_robust', {}).get('ate'),
            'bias_magnitude': None,
            'statistical_significance': None,
            'business_impact': lift_metrics
        },
        'method_comparison': {},
        'robustness_assessment': 'completed' if 'robustness' in results else 'failed',
        'sensitivity_assessment': 'completed' if 'sensitivity' in results else 'failed'
    }
    
    # Add method comparison
    for method, result in results.items():
        if method != 'naive' and 'ate' in result:
            summary['method_comparison'][method] = {
                'ate': result['ate'],
                'confidence_interval': [
                    result.get('ate_ci_lower'), 
                    result.get('ate_ci_upper')
                ]
            }
    
    # Calculate bias magnitude
    if 'doubly_robust' in results:
        bias = results['naive']['ate'] - results['doubly_robust']['ate']
        summary['executive_summary']['bias_magnitude'] = bias
        summary['executive_summary']['statistical_significance'] = results['doubly_robust'].get('ate_se', 0) > 0
    
    return summary

if __name__ == "__main__":
    run_pipeline()
