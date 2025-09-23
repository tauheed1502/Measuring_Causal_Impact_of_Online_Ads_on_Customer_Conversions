import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

def plot_results(results_dict: Dict, save_path: Optional[str] = None):
    """Create comprehensive plots for causal analysis results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Causal Analysis Results', fontsize=16)
    
    # Plot 1: Treatment Effect Comparison
    plot_treatment_effects(results_dict, axes[0, 0])
    
    # Plot 2: Propensity Score Distribution
    if 'propensity_scores' in results_dict:
        plot_propensity_distribution(results_dict['propensity_scores'], axes[0, 1])
    
    # Plot 3: Robustness Check
    if 'robustness' in results_dict:
        plot_robustness_results(results_dict['robustness'], axes[1, 0])
    
    # Plot 4: Sensitivity Analysis
    if 'sensitivity' in results_dict:
        plot_sensitivity_analysis(results_dict['sensitivity'], axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_treatment_effects(results: Dict, ax):
    """Plot comparison of treatment effects from different methods"""
    
    methods = ['naive', 'psm', 'doubly_robust', 'causal_forest']
    effects = []
    errors = []
    method_names = []
    
    for method in methods:
        if method in results:
            effects.append(results[method]['ate'])
            errors.append(results[method].get('ate_se', 0))
            method_names.append(method.replace('_', ' ').title())
    
    if effects:
        bars = ax.bar(method_names, effects, yerr=errors, capsize=5, 
                     color=['red', 'blue', 'green', 'orange'][:len(effects)], alpha=0.7)
        
        ax.set_ylabel('Treatment Effect (ATE)')
        ax.set_title('Treatment Effect Estimates by Method')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar, effect in zip(bars, effects):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{effect:.4f}', ha='center', va='bottom')
    
    ax.tick_params(axis='x', rotation=45)

def plot_propensity_distribution(df: pd.DataFrame, ax, treatment_col: str = 'treatment'):
    """Plot propensity score distributions for treatment and control groups"""
    
    if 'propensity_score' in df.columns:
        treated = df[df[treatment_col] == 1]['propensity_score']
        control = df[df[treatment_col] == 0]['propensity_score']
        
        ax.hist(control, bins=30, alpha=0.7, label='Control', color='blue', density=True)
        ax.hist(treated, bins=30, alpha=0.7, label='Treated', color='red', density=True)
        
        ax.set_xlabel('Propensity Score')
        ax.set_ylabel('Density')
        ax.set_title('Propensity Score Distribution')
        ax.legend()
        
        # Add overlap statistics
        overlap = min(treated.max(), control.max()) - max(treated.min(), control.min())
        ax.text(0.05, 0.95, f'Overlap: {overlap:.3f}', transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

def plot_robustness_results(robustness_results: Dict, ax):
    """Plot robustness check results"""
    
    if 'subset_validation' in robustness_results:
        subset_results = robustness_results['subset_validation']
        
        subsets = []
        estimates = []
        errors = []
        
        for subset_name, result in subset_results.items():
            if 'ate' in result:
                subsets.append(subset_name.replace('_', ' ').title())
                estimates.append(result['ate'])
                errors.append(result.get('ate_se', 0))
        
        if estimates:
            ax.errorbar(range(len(subsets)), estimates, yerr=errors, 
                       marker='o', capsize=5, linestyle='-', linewidth=2)
            
            ax.set_xticks(range(len(subsets)))
            ax.set_xticklabels(subsets, rotation=45)
            ax.set_ylabel('Treatment Effect (ATE)')
            ax.set_title('Robustness Across Subsets')
            ax.axhline(y=np.mean(estimates), color='red', linestyle='--', alpha=0.7, label='Mean')
            ax.legend()

def plot_sensitivity_analysis(sensitivity_results: Dict, ax):
    """Plot sensitivity analysis results"""
    
    if 'rosenbaum_bounds' in sensitivity_results:
        bounds = sensitivity_results['rosenbaum_bounds']['bounds']
        
        gammas = [b['gamma'] for b in bounds]
        p_upper = [b['p_value_upper'] for b in bounds]
        p_lower = [b['p_value_lower'] for b in bounds]
        
        ax.plot(gammas, p_upper, 'r-', label='Upper Bound', linewidth=2)
        ax.plot(gammas, p_lower, 'b-', label='Lower Bound', linewidth=2)
        ax.axhline(y=0.05, color='black', linestyle='--', alpha=0.7, label='α = 0.05')
        
        ax.set_xlabel('Γ (Hidden Bias Factor)')
        ax.set_ylabel('P-value')
        ax.set_title('Rosenbaum Sensitivity Bounds')
        ax.legend()
        ax.grid(True, alpha=0.3)

def plot_covariate_balance(df: pd.DataFrame, 
                          confounders: List[str],
                          treatment_col: str = 'treatment',
                          save_path: Optional[str] = None):
    """Plot covariate balance before and after matching"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Covariate Balance Assessment', fontsize=16)
    
    # Standardized mean differences
    plot_smd_comparison(df, confounders, treatment_col, axes[0, 0])
    
    # Propensity score balance
    if 'propensity_score' in df.columns:
        plot_propensity_balance(df, treatment_col, axes[0, 1])
    
    # Distribution comparison for key variables
    plot_covariate_distributions(df, confounders[:4], treatment_col, axes[1, :])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_smd_comparison(df: pd.DataFrame, confounders: List[str], treatment_col: str, ax):
    """Plot standardized mean differences"""
    
    from src.data.preprocessing import balance_check
    
    balance_stats = balance_check(df, treatment_col)
    
    variables = balance_stats['variable'][:10]  # Top 10 variables
    smds = balance_stats['standardized_mean_diff'][:10]
    
    colors = ['red' if abs(smd) > 0.1 else 'green' for smd in smds]
    
    bars = ax.barh(variables, smds, color=colors, alpha=0.7)
    ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='Imbalance Threshold')
    ax.axvline(x=-0.1, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    ax.set_xlabel('Standardized Mean Difference')
    ax.set_title('Covariate Balance')
    ax.legend()

def plot_heterogeneous_effects(treatment_effects: np.ndarray,
                             df: pd.DataFrame,
                             key_variables: List[str],
                             save_path: Optional[str] = None):
    """Plot heterogeneous treatment effects"""
    
    n_vars = len(key_variables)
    fig, axes = plt.subplots(1, min(n_vars, 3), figsize=(15, 5))
    if n_vars == 1:
        axes = [axes]
    
    for i, var in enumerate(key_variables[:3]):
        if var in df.columns:
            # Create quartiles
            df_plot = df.copy()
            df_plot['treatment_effect'] = treatment_effects
            df_plot[f'{var}_quartile'] = pd.qcut(df_plot[var], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            # Plot by quartile
            quartile_effects = df_plot.groupby(f'{var}_quartile')['treatment_effect'].agg(['mean', 'std'])
            
            axes[i].bar(quartile_effects.index, quartile_effects['mean'], 
                       yerr=quartile_effects['std'], capsize=5, alpha=0.7)
            axes[i].set_title(f'Treatment Effects by {var.title()} Quartile')
            axes[i].set_ylabel('Treatment Effect')
            axes[i].axhline(y=treatment_effects.mean(), color='red', linestyle='--', alpha=0.7, label='Overall ATE')
            axes[i].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
