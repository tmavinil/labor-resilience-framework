import pandas as pd

class ResilienceScorer:
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the scorer with industry-specific data.
        The data DataFrame must include the columns:
        - 'Is_Absent' (int or bool)
        - 'Overtime_Hrs' (float)
        - 'Scheduled_Hrs' (float)
        - 'Terminated' (int or bool)
        Optional columns:
        - 'Focus_Efficiency'
        - 'National_Unemployment_Rate'
        """
        self.data = data

    def get_shap_dataframe(self) -> pd.DataFrame:
        """
        Calculates the underlying metrics needed for the resilience score and returns 
        them as a clean Pandas DataFrame. This preserves manager-friendly feature names
        so that SHAP plots are highly readable for non-technical audiences.
        """
        # Absenteeism Rate
        absenteeism_rate = self.data['Is_Absent'].mean() if 'Is_Absent' in self.data.columns else 0.0
        
        # Overtime Dependency
        total_overtime = self.data['Overtime_Hrs'].sum() if 'Overtime_Hrs' in self.data.columns else 0.0
        total_scheduled = self.data['Scheduled_Hrs'].sum() if 'Scheduled_Hrs' in self.data.columns else 0.0
        overtime_dependency = total_overtime / total_scheduled if total_scheduled > 0 else 0.0
        
        # Attrition Rate
        attrition_rate = self.data['Terminated'].mean() if 'Terminated' in self.data.columns else 0.0
        
        # Focus Efficiency
        focus_efficiency = self.data['Focus_Efficiency'].mean() if 'Focus_Efficiency' in self.data.columns else 0.60
        
        # National Unemployment Rate (Macro-economic context)
        unemployment_rate = self.data['National_Unemployment_Rate'].mean() if 'National_Unemployment_Rate' in self.data.columns else 4.0

        # Construct a DataFrame with clean, readable feature names
        df = pd.DataFrame([{
            'Absenteeism_Rate': absenteeism_rate,
            'Overtime_Dependency': overtime_dependency,
            'Attrition_Rate': attrition_rate,
            'Focus_Efficiency': focus_efficiency,
            'National_Unemployment_Rate': unemployment_rate
        }])
        
        return df

    def calculate_metrics(self) -> dict:
        """
        Maintains backward compatibility by returning the first row of the 
        SHAP DataFrame as a dictionary.
        """
        df = self.get_shap_dataframe()
        return df.iloc[0].to_dict()

    def compute_score(self) -> float:
        """
        Calculates a composite 'Resilience Score' from 0 to 100 based on the SHAP DataFrame.
        """
        df = self.get_shap_dataframe()
        metrics = df.iloc[0]
        
        # Normalize the metrics using assumed domain-specific worst-case thresholds.
        norm_absenteeism = min(metrics['Absenteeism_Rate'] / 0.20, 1.0)
        norm_overtime = min(metrics['Overtime_Dependency'] / 0.30, 1.0)
        norm_attrition = min(metrics['Attrition_Rate'] / 0.10, 1.0)
        
        # Penalize the resilience score
        penalty = (norm_absenteeism * 30) + (norm_overtime * 30) + (norm_attrition * 40)
        
        # Buffers from Focus Efficiency and Unemployment (optional logic enhancements)
        focus_buffer = max((metrics['Focus_Efficiency'] - 0.60) * 10.0, 0.0)
        macro_buffer = max((metrics['National_Unemployment_Rate'] - 4.0) * 2.0, 0.0)
        
        resilience_score = max(min(100 - penalty + focus_buffer + macro_buffer, 100.0), 0.0)
        
        return round(resilience_score, 2)

    def categorize_risk(self, score: float) -> str:
        """
        Categorizes the resilience score into risk levels.
        """
        if score >= 75:
            return 'Low'
        elif score >= 50:
            return 'Moderate'
        else:
            return 'High'

    def evaluate(self) -> dict:
        """
        Evaluates the data, calculates the score, and categorizes the risk.
        Returns a dictionary with all results.
        """
        score = self.compute_score()
        risk_category = self.categorize_risk(score)
        metrics_df = self.get_shap_dataframe()
        
        return {
            'resilience_score': score,
            'risk_category': risk_category,
            'metrics_df': metrics_df, # Returned for SHAP directly
            'metrics_dict': metrics_df.iloc[0].to_dict()
        }
