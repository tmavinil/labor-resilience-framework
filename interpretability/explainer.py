import shap
import pandas as pd
import numpy as np

class ResilienceExplainer:
    def __init__(self, background_metrics: pd.DataFrame):
        """
        Initializes the SHAP explainer for the ResilienceScorer.
        
        :param background_metrics: A pandas DataFrame of historical/representative metrics 
                                   with columns: ['absenteeism_rate', 'overtime_dependency', 'attrition_rate']
                                   This is used to establish the baseline 'expected' score for SHAP.
        """
        self.background_metrics = background_metrics
        
        # Initialize the SHAP Explainer with a wrapper function that mirrors ResilienceScorer's math
        self.explainer = shap.Explainer(self._predict_wrapper, self.background_metrics)

    def _predict_wrapper(self, X: np.ndarray) -> np.ndarray:
        """
        A scoring wrapper function that mimicking the ResilienceScorer to calculate
        the scores for a batch of metrics. SHAP passes 2D arrays to this function.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        scores = []
        for row in X:
            absenteeism_rate, overtime_dependency, attrition_rate = row
            
            # Constraints and thresholds from ResilienceScorer
            norm_absenteeism = min(absenteeism_rate / 0.20, 1.0)
            norm_overtime = min(overtime_dependency / 0.30, 1.0)
            norm_attrition = min(attrition_rate / 0.10, 1.0)
            
            # Calculate penalties based on weights
            penalty = (norm_absenteeism * 30) + (norm_overtime * 30) + (norm_attrition * 40)
            score = max(100 - penalty, 0.0)
            scores.append(round(score, 2))
            
        return np.array(scores)

    def explain_score(self, current_metrics: dict) -> dict:
        """
        Calculates SHAP values to explain which factor most impacted the final Resilience Score.
        
        :param current_metrics: A dict from ResilienceScorer.calculate_metrics()
        :return: A dictionary containing the SHAP values and the top impacting factor.
        """
        # Format the instance into a DataFrame ensuring the columns match the background data
        df_instance = pd.DataFrame([current_metrics], columns=['absenteeism_rate', 'overtime_dependency', 'attrition_rate'])
        
        # Compute SHAP values
        shap_values = self.explainer(df_instance)
        
        # The impact values for this specific instance (index 0 since we only passed one row)
        impacts = shap_values.values[0]
        feature_names = shap_values.feature_names
        
        # Find the feature that had the most negative impact (i.e. lowered the score the most)
        most_negative_idx = np.argmin(impacts)
        top_negative_factor = feature_names[most_negative_idx]
        top_negative_impact = impacts[most_negative_idx]
        
        # Structure the explanation results
        explanation_result = {
            'shap_values': dict(zip(feature_names, impacts)),
            'base_value': shap_values.base_values[0],
            'top_negative_factor': top_negative_factor,
            'top_negative_impact': round(top_negative_impact, 2)
        }
        
        return explanation_result
