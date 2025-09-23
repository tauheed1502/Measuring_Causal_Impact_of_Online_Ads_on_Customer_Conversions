import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Tuple

class PropensityScoreModel:
    def __init__(self, model_type: str = 'logistic'):
        self.model_type = model_type
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.propensity_scores = None
        
    def train_propensity_model(self, X: pd.DataFrame, treatment: pd.Series) -> Dict:
        """Train propensity score model"""
        # Fit the model
        self.model.fit(X, treatment)
        
        # Calculate propensity scores
        self.propensity_scores = self.model.predict_proba(X)[:, 1]
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, treatment, cv=5, scoring='roc_auc')
        
        results = {
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'propensity_scores': self.propensity_scores
        }
        
        print(f"Propensity model trained. CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        return results
    
    def check_overlap(self) -> Dict:
        """Check overlap in propensity scores between treatment groups"""
        if self.propensity_scores is None:
            raise ValueError("Model must be trained first")
            
        # Calculate overlap statistics
        min_ps = self.propensity_scores.min()
        max_ps = self.propensity_scores.max()
        
        # Check for extreme propensity scores
        extreme_low = (self.propensity_scores < 0.1).sum()
        extreme_high = (self.propensity_scores > 0.9).sum()
        
        return {
            'min_propensity': min_ps,
            'max_propensity': max_ps,
            'extreme_low_count': extreme_low,
            'extreme_high_count': extreme_high,
            'overlap_ok': (min_ps > 0.01) and (max_ps < 0.99)
        }

def propensity_score_matching(df: pd.DataFrame, 
                            confounders: list,
                            treatment_col: str = 'treatment',
                            outcome_col: str = 'conversion',
                            caliper: float = 0.1) -> Tuple[pd.DataFrame, Dict]:
    """Perform 1:1 propensity score matching"""
    
    # Train propensity model
    ps_model = PropensityScoreModel('logistic')
    results = ps_model.train_propensity_model(df[confounders], df[treatment_col])
    
    # Add propensity scores to dataframe
    df_with_ps = df.copy()
    df_with_ps['propensity_score'] = results['propensity_scores']
    
    # Separate treatment and control groups
    treated = df_with_ps[df_with_ps[treatment_col] == 1].copy()
    control = df_with_ps[df_with_ps[treatment_col] == 0].copy()
    
    # Nearest neighbor matching
    nn_model = NearestNeighbors(n_neighbors=1, metric='manhattan')
    nn_model.fit(control[['propensity_score']])
    
    # Find matches for each treated unit
    distances, indices = nn_model.kneighbors(treated[['propensity_score']])
    
    # Apply caliper constraint
    valid_matches = distances.flatten() <= caliper
    matched_treated = treated[valid_matches].copy()
    matched_control = control.iloc[indices.flatten()[valid_matches]].copy()
    
    # Combine matched data
    matched_data = pd.concat([matched_treated, matched_control], ignore_index=True)
    
    # Calculate treatment effect
    treated_outcome = matched_treated[outcome_col].mean()
    control_outcome = matched_control[outcome_col].mean()
    ate = treated_outcome - control_outcome
    
    matching_results = {
        'ate': ate,
        'treated_outcome': treated_outcome,
        'control_outcome': control_outcome,
        'matched_pairs': len(matched_treated),
        'total_treated': len(treated),
        'matching_rate': len(matched_treated) / len(treated)
    }
    
    print(f"PSM Results - ATE: {ate:.4f}, Matched pairs: {len(matched_treated)}")
    return matched_data, matching_results
