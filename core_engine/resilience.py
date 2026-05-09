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
        """
        self.data = data

    def calculate_metrics(self) -> dict:
        """
        Calculates the underlying metrics needed for the resilience score.
        """
        # Absenteeism Rate: Total absent days / Total records
        absenteeism_rate = self.data['Is_Absent'].mean()
        
        # Overtime Dependency: Total Overtime Hours / Total Scheduled Hours
        total_overtime = self.data['Overtime_Hrs'].sum()
        total_scheduled = self.data['Scheduled_Hrs'].sum()
        overtime_dependency = total_overtime / total_scheduled if total_scheduled > 0 else 0
        
        # Attrition Rate: Total terminations / Total records
        attrition_rate = self.data['Terminated'].mean()
        
        return {
            'absenteeism_rate': absenteeism_rate,
            'overtime_dependency': overtime_dependency,
            'attrition_rate': attrition_rate
        }

    def compute_score(self) -> float:
        """
        Calculates a composite 'Resilience Score' from 0 to 100.
        A higher score means higher resilience (lower risk).
        """
        metrics = self.calculate_metrics()
        
        # Normalize the metrics using assumed domain-specific worst-case thresholds.
        # Max expected absenteeism ~ 20%
        # Max expected overtime dependency ~ 30%
        # Max expected attrition ~ 10%
        norm_absenteeism = min(metrics['absenteeism_rate'] / 0.20, 1.0)
        norm_overtime = min(metrics['overtime_dependency'] / 0.30, 1.0)
        norm_attrition = min(metrics['attrition_rate'] / 0.10, 1.0)
        
        # Penalize the resilience score based on these negative metrics.
        # Base score is 100. We distribute penalties with the following weights:
        # Absenteeism: 30%, Overtime Dependency: 30%, Attrition: 40%
        penalty = (norm_absenteeism * 30) + (norm_overtime * 30) + (norm_attrition * 40)
        resilience_score = max(100 - penalty, 0.0)
        
        return round(resilience_score, 2)

    def categorize_risk(self, score: float) -> str:
        """
        Categorizes the resilience score into risk levels.
        - Low Risk: Score >= 75
        - Moderate Risk: 50 <= Score < 75
        - High Risk: Score < 50
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
        metrics = self.calculate_metrics()
        score = self.compute_score()
        risk_category = self.categorize_risk(score)
        
        return {
            'resilience_score': score,
            'risk_category': risk_category,
            'metrics': metrics
        }
