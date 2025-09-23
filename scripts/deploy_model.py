import pickle
import pandas as pd
import numpy as np
from typing import Dict, List
import joblib
import yaml
from datetime import datetime

class CausalAdModel:
    """Production-ready causal inference model for ad impact measurement"""
    
    def __init__(self, model_path: str = 'results/models/'):
        self.model_path = model_path
        self.propensity_model = None
        self.outcome_models = {}
        self.confounders = []
        self.scaler = None
        self.deployed_at = None
        
    def deploy_model(self, results: Dict):
        """Deploy trained causal models for production use"""
        
        print("🚀 Deploying Causal Ad Impact Model")
        
        # Extract and save models
        if 'doubly_robust' in results:
            self._save_doubly_robust_components(results['doubly_robust'])
        
        if 'causal_forest' in results:
            self._save_causal_forest(results['causal_forest'])
        
        # Save metadata
        self._save_deployment_metadata(results)
        
        self.deployed_at = datetime.now()
        print(f"✅ Model deployed successfully at {self.deployed_at}")
        
    def _save_doubly_robust_components(self, dr_results: Dict):
        """Save doubly robust model components"""
        
        # Save propensity scores and models (would need actual model objects)
        deployment_package = {
            'propensity_scores': dr_results.get('propensity_scores'),
            'mu0': dr_results.get('mu0'),  # Control outcome model predictions
            'mu1': dr_results.get('mu1'),  # Treatment outcome model predictions
            'ate': dr_results.get('ate'),
            'ate_se': dr_results.get('ate_se')
        }
        
        with open(f'{self.model_path}doubly_robust_model.pkl', 'wb') as f:
            pickle.dump(deployment_package, f)
            
        print("✅ Doubly robust components saved")
    
    def _save_causal_forest(self, cf_results: Dict):
        """Save causal forest model"""
        
        if 'model' in cf_results:
            # Save the actual model
            joblib.dump(cf_results['model'], f'{self.model_path}causal_forest_model.pkl')
            
            # Save treatment effects and metadata
            cf_package = {
                'treatment_effects': cf_results.get('treatment_effects'),
                'ate': cf_results.get('ate'),
                'ate_se': cf_results.get('ate_se'),
                'effect_variance': cf_results.get('effect_variance')
            }
            
            with open(f'{self.model_path}causal_forest_metadata.pkl', 'wb') as f:
                pickle.dump(cf_package, f)
                
            print("✅ Causal forest model saved")
    
    def _save_deployment_metadata(self, results: Dict):
        """Save deployment metadata and configuration"""
        
        metadata = {
            'deployment_timestamp': datetime.now().isoformat(),
            'model_version': '1.0.0',
            'training_summary': {
                'methods_used': list(results.keys()),
                'primary_estimate': results.get('doubly_robust', {}).get('ate'),
                'confidence_interval': [
                    results.get('doubly_robust', {}).get('ate_ci_lower'),
                    results.get('doubly_robust', {}).get('ate_ci_upper')
                ]
            },
            'validation_status': {
                'robustness_checks': 'robustness' in results,
                'sensitivity_analysis': 'sensitivity' in results,
                'placebo_tests': 'placebo' in results
            }
        }
        
        with open(f'{self.model_path}deployment_metadata.yaml', 'w') as f:
            yaml.dump(metadata, f)
        
        print("✅ Deployment metadata saved")

def predict_treatment_effect(user_features: pd.DataFrame, 
                           model_path: str = 'results/models/') -> Dict:
    """Predict treatment effect for new users"""
    
    try:
        # Load causal forest model
        cf_model = joblib.load(f'{model_path}causal_forest_model.pkl')
        
        # Predict individual treatment effects
        individual_effects = cf_model.effect(user_features.values)
        
        # Load doubly robust estimates for comparison
        with open(f'{model_path}doubly_robust_model.pkl', 'rb') as f:
            dr_model = pickle.load(f)
        
        population_ate = dr_model['ate']
        
        return {
            'individual_effects': individual_effects,
            'mean_effect': np.mean(individual_effects),
            'population_ate': population_ate,
            'high_responders': user_features[individual_effects > np.percentile(individual_effects, 75)],
            'low_responders': user_features[individual_effects < np.percentile(individual_effects, 25)]
        }
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        return {'error': str(e)}

def create_targeting_recommendations(prediction_results: Dict) -> Dict:
    """Generate targeting recommendations based on predicted effects"""
    
    if 'error' in prediction_results:
        return prediction_results
    
    individual_effects = prediction_results['individual_effects']
    
    # Targeting thresholds
    high_value_threshold = np.percentile(individual_effects, 80)
    medium_value_threshold = np.percentile(individual_effects, 60)
    
    recommendations = {
        'high_value_segment': {
            'threshold': high_value_threshold,
            'expected_lift': np.mean(individual_effects[individual_effects >= high_value_threshold]),
            'recommended_action': 'Aggressive targeting with premium creatives'
        },
        'medium_value_segment': {
            'threshold': medium_value_threshold,
            'expected_lift': np.mean(individual_effects[(individual_effects >= medium_value_threshold) & 
                                                       (individual_effects < high_value_threshold)]),
            'recommended_action': 'Standard targeting with optimized frequency'
        },
        'low_value_segment': {
            'threshold': 0,
            'expected_lift': np.mean(individual_effects[individual_effects < medium_value_threshold]),
            'recommended_action': 'Limited targeting or exclude from campaigns'
        }
    }
    
    return recommendations

def monitoring_dashboard_data(model_path: str = 'results/models/') -> Dict:
    """Generate data for monitoring dashboard"""
    
    try:
        # Load deployment metadata
        with open(f'{model_path}deployment_metadata.yaml', 'r') as f:
            metadata = yaml.safe_load(f)
        
        # Load model performance
        with open(f'{model_path}doubly_robust_model.pkl', 'rb') as f:
            dr_model = pickle.load(f)
        
        dashboard_data = {
            'model_health': {
                'deployment_date': metadata['deployment_timestamp'],
                'model_version': metadata['model_version'],
                'status': 'healthy'  # Would be computed based on recent predictions
            },
            'performance_metrics': {
                'current_ate': dr_model['ate'],
                'confidence_interval': [
                    dr_model.get('ate_ci_lower', dr_model['ate'] - 1.96 * dr_model['ate_se']),
                    dr_model.get('ate_ci_upper', dr_model['ate'] + 1.96 * dr_model['ate_se'])
                ],
                'precision': dr_model['ate_se']
            },
            'usage_stats': {
                'predictions_made': 'N/A',  # Would track actual usage
                'last_prediction': 'N/A',
                'average_daily_usage': 'N/A'
            }
        }
        
        return dashboard_data
        
    except Exception as e:
        return {'error': f'Dashboard data generation failed: {str(e)}'}

if __name__ == "__main__":
    # Example deployment workflow
    
    # Load results from training
    with open('results/causal_analysis_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    # Deploy model
    model = CausalAdModel()
    model.deploy_model(results)
    
    print("🔧 Model deployment completed!")
    print("📊 Use monitoring_dashboard_data() to track model performance")
    print("🎯 Use predict_treatment_effect() for new predictions")
