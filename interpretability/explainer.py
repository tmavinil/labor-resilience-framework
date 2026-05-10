import shap
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class ResilienceExplainer:
    def __init__(self, background_metrics: pd.DataFrame):
        """
        Initializes the SHAP explainer for the ResilienceScorer.
        
        :param background_metrics: A pandas DataFrame of historical/representative metrics 
                                   This is used to establish the baseline 'expected' score for SHAP.
                                   Expected columns: 'Absenteeism_Rate', 'Overtime_Dependency', 
                                   'Attrition_Rate', 'Focus_Efficiency', 'National_Unemployment_Rate'
        """
        # Define the exact friendly names we expect from the ResilienceScorer's get_shap_dataframe()
        self.feature_columns = [
            'Absenteeism_Rate', 
            'Overtime_Dependency', 
            'Attrition_Rate', 
            'Focus_Efficiency', 
            'National_Unemployment_Rate'
        ]
        
        # Ensure all columns exist in the background data, providing safe defaults if missing
        for col in self.feature_columns:
            if col not in background_metrics.columns:
                if col == 'Focus_Efficiency':
                    background_metrics[col] = 0.60
                elif col == 'National_Unemployment_Rate':
                    background_metrics[col] = 4.0
                else:
                    background_metrics[col] = 0.0
                    
        # Reorder columns to ensure consistent matrix shapes
        self.background_metrics = background_metrics[self.feature_columns]
        
        # Initialize the SHAP Explainer with the wrapper function
        self.explainer = shap.Explainer(self._predict_wrapper, self.background_metrics)

    def _predict_wrapper(self, X: np.ndarray) -> np.ndarray:
        """
        A scoring wrapper function mimicking the ResilienceScorer to calculate
        the scores for a batch of metrics. SHAP passes 2D arrays to this function.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        scores = []
        for row in X:
            absenteeism_rate, overtime_dependency, attrition_rate, focus_efficiency, nat_unemployment = row
            
            # Constraints and thresholds from ResilienceScorer
            norm_absenteeism = min(absenteeism_rate / 0.20, 1.0)
            norm_overtime = min(overtime_dependency / 0.30, 1.0)
            norm_attrition = min(attrition_rate / 0.10, 1.0)
            
            # Calculate penalties based on weights
            penalty = (norm_absenteeism * 30) + (norm_overtime * 30) + (norm_attrition * 40)
            
            # Buffers
            focus_buffer = max((focus_efficiency - 0.60) * 10.0, 0.0)
            macro_buffer = max((nat_unemployment - 4.0) * 2.0, 0.0)
            
            score = max(min(100 - penalty + focus_buffer + macro_buffer, 100.0), 0.0)
            scores.append(round(score, 2))
            
        return np.array(scores)

    def explain_score(self, current_metrics_df: pd.DataFrame) -> dict:
        """
        Calculates SHAP values to explain which factor most impacted the final Resilience Score.
        
        :param current_metrics_df: A pandas DataFrame containing the metrics (e.g. from get_shap_dataframe())
        :return: A dictionary containing the SHAP values, top negative factor, and a text narrative.
        """
        # Ensure it has all columns
        df_instance = current_metrics_df.copy()
        for col in self.feature_columns:
            if col not in df_instance.columns:
                if col == 'Focus_Efficiency':
                    df_instance[col] = 0.60
                elif col == 'National_Unemployment_Rate':
                    df_instance[col] = 4.0
                else:
                    df_instance[col] = 0.0
                    
        df_instance = df_instance[self.feature_columns]
        
        # Compute SHAP values
        shap_values = self.explainer(df_instance)
        
        # The impact values for this specific instance (index 0 since we only passed one row)
        impacts = shap_values.values[0]
        feature_names = shap_values.feature_names
        
        # Find the feature that had the most negative impact (i.e. lowered the score the most)
        most_negative_idx = np.argmin(impacts)
        top_negative_factor = feature_names[most_negative_idx]
        top_negative_impact = impacts[most_negative_idx]
        
        # Calculate narrative summary
        shap_dict = dict(zip(feature_names, impacts))
        narrative = self.generate_explanation_summary(shap_dict)
        
        # Structure the explanation results
        explanation_result = {
            'shap_values': shap_dict,
            'base_value': shap_values.base_values[0],
            'top_negative_factor': top_negative_factor,
            'top_negative_impact': round(top_negative_impact, 2),
            'narrative_summary': narrative
        }
        
        return explanation_result

    def generate_explanation_summary(self, shap_dict: dict) -> str:
        """
        Generates a human-readable text narrative explaining the SHAP values.
        """
        # Map raw feature names to even friendlier string if necessary, but we are already using good names
        friendly_names = {
            'Absenteeism_Rate': 'Absenteeism',
            'Overtime_Dependency': 'Overtime Dependency',
            'Attrition_Rate': 'Attrition',
            'Focus_Efficiency': 'Focus Efficiency',
            'National_Unemployment_Rate': 'National Unemployment Rate'
        }
        
        # Find the primary negative driver
        negative_factors = {k: v for k, v in shap_dict.items() if v < 0}
        if negative_factors:
            primary_risk_key = min(negative_factors, key=negative_factors.get)
            primary_risk_impact = round(negative_factors[primary_risk_key], 1)
            primary_risk_name = friendly_names.get(primary_risk_key, primary_risk_key)
            risk_sentence = f"The primary risk factor is {primary_risk_name}, contributing {primary_risk_impact} points to the score"
        else:
            risk_sentence = "There are no significant negative risk factors pulling the score down"
            
        # Analyze buffers (positive impacts outside of base values)
        # Assuming Focus Efficiency and National Unemployment can act as buffers
        buffer_impacts = [
            (friendly_names.get(k, k), round(v, 1)) 
            for k, v in shap_dict.items() 
            if v > 0 and k in ['Focus_Efficiency', 'National_Unemployment_Rate']
        ]
        
        if buffer_impacts:
            buffer_texts = [f"the {name} added a +{val} point stability buffer" for name, val in buffer_impacts]
            macro_sentence = ", while " + " and ".join(buffer_texts) + "."
        else:
            macro_sentence = "."
            
        return risk_sentence + macro_sentence

    def save_visual_explanation(self, data: pd.DataFrame, output_path: str = 'examples/resilience_risk_factors.png'):
        """
        Generates and saves a SHAP summary plot for the given data to a PNG file.
        
        :param data: A pandas DataFrame of metrics to explain.
        :param output_path: The path where the plot image will be saved.
        """
        df_instance = data.copy()
        
        # Ensure it has all required columns
        for col in self.feature_columns:
            if col not in df_instance.columns:
                if col == 'Focus_Efficiency':
                    df_instance[col] = 0.60
                elif col == 'National_Unemployment_Rate':
                    df_instance[col] = 4.0
                else:
                    df_instance[col] = 0.0
                    
        df_instance = df_instance[self.feature_columns]
        
        # Compute SHAP values
        shap_values = self.explainer(df_instance)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate summary plot without displaying it to screen
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, df_instance, show=False)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Visual explanation saved successfully to {output_path}")

    def explain_trends(self, time_series_df: pd.DataFrame) -> dict:
        """
        Calculates SHAP values for each period in a time-series dataset and
        performs a 'Driver Drift' analysis to identify worsening risk factors.
        
        :param time_series_df: A pandas DataFrame containing metrics over time. Must be sorted chronologically.
        :return: A dictionary with trend analysis and a narrative summary.
        """
        # Ensure it has all columns
        df_instance = time_series_df.copy()
        
        for col in self.feature_columns:
            if col not in df_instance.columns:
                if col == 'Focus_Efficiency':
                    df_instance[col] = 0.60
                elif col == 'National_Unemployment_Rate':
                    df_instance[col] = 4.0
                else:
                    df_instance[col] = 0.0
                    
        eval_df = df_instance[self.feature_columns]
        
        # Compute SHAP values for the entire time series
        shap_values = self.explainer(eval_df)
        
        # If less than 2 periods, we can't calculate a trend
        if len(eval_df) < 2:
            return {"error": "Not enough data points to calculate trends. Minimum 2 periods required."}
            
        # Compare the first available period to the last to find "drift"
        first_period_impacts = shap_values.values[0]
        last_period_impacts = shap_values.values[-1]
        
        feature_names = shap_values.feature_names
        
        drift_results = {}
        worst_drift_feature = None
        worst_drift_value = 0.0 # Negative means it got "worse" (more negative impact)
        
        friendly_names = {
            'Absenteeism_Rate': 'Absenteeism',
            'Overtime_Dependency': 'Overtime',
            'Attrition_Rate': 'Attrition',
            'Focus_Efficiency': 'Focus Efficiency',
            'National_Unemployment_Rate': 'National Unemployment'
        }
        
        for i, feature in enumerate(feature_names):
            # Calculate drift (how much more negative the impact became)
            impact_change = last_period_impacts[i] - first_period_impacts[i]
            drift_results[feature] = impact_change
            
            # We are looking for the factor that worsened the most (most negative change)
            if impact_change < worst_drift_value:
                worst_drift_value = impact_change
                worst_drift_feature = feature
                
        if worst_drift_feature and worst_drift_value < -0.01:
            friendly_worst = friendly_names.get(worst_drift_feature, worst_drift_feature)
            # Absolute points worsened
            worsened_by = abs(round(worst_drift_value, 1))
            
            # Check unemployment trend for context
            unemp_idx = feature_names.index('National_Unemployment_Rate') if 'National_Unemployment_Rate' in feature_names else -1
            unemp_drift = 0
            if unemp_idx != -1:
                unemp_drift = last_period_impacts[unemp_idx] - first_period_impacts[unemp_idx]
            
            context = ""
            if abs(unemp_drift) < 1.0:
                context = " despite stable National Unemployment levels"
            elif unemp_drift < -1.0:
                context = " exacerbated by worsening National Unemployment levels"
            else:
                context = " offset partially by improving National Unemployment levels"
                
            if worst_drift_feature == 'Overtime_Dependency':
                narrative = f"Over the analyzed period, the impact of Overtime on the Resilience Score has worsened by {worsened_by} points, indicating a trend toward systemic burnout{context}."
            else:
                narrative = f"Over the analyzed period, the impact of {friendly_worst} on the Resilience Score has worsened by {worsened_by} points, indicating a trend toward systemic risk{context}."
                
        else:
            narrative = "Over the analyzed period, no significant risk factors showed a worsening trend."

        return {
            "drift_results": drift_results,
            "narrative_summary": narrative,
            "shap_values": shap_values
        }

    def plot_trend_evolution(self, shap_values, feature_names, output_path: str = 'examples/resilience_trend_plot.png'):
        """
        Generates and saves a multi-line chart showing the evolution of SHAP values over time.
        
        :param shap_values: The SHAP values object or array containing the impacts.
        :param feature_names: List of feature names corresponding to the columns in shap_values.
        :param output_path: Path where the PNG will be saved.
        """
        # Extract values matrix
        if hasattr(shap_values, 'values'):
            impacts = shap_values.values
        else:
            impacts = np.array(shap_values)
            
        # Limit to the last 12 months for the plot
        if len(impacts) > 12:
            impacts = impacts[-12:]
            
        periods = range(1, len(impacts) + 1)
        
        plt.figure(figsize=(12, 7))
        
        friendly_names = {
            'Absenteeism_Rate': 'Absenteeism',
            'Overtime_Dependency': 'Overtime',
            'Attrition_Rate': 'Attrition',
            'Focus_Efficiency': 'Focus Efficiency',
            'National_Unemployment_Rate': 'FRED Multiplier'
        }
        
        for i, feature in enumerate(feature_names):
            feature_impacts = impacts[:, i]
            label = friendly_names.get(feature, feature)
            plt.plot(periods, feature_impacts, marker='o', linewidth=2, label=label)
            
        plt.title('Workforce Resilience Risk Factor Evolution (12-Month Trend)', fontsize=14, pad=15)
        plt.xlabel('Time Period (Last 12 Months)', fontsize=12)
        plt.ylabel('Impact on Resilience Score (SHAP Value)', fontsize=12)
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # Place legend outside the plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Trend evolution plot saved successfully to {output_path}")
