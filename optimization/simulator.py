import pandas as pd
import numpy as np

class WorkforceSimulator:
    def __init__(self, base_data: pd.DataFrame, scorer_class):
        """
        Initializes the simulator with a base dataset and the ResilienceScorer class.
        
        :param base_data: A pandas DataFrame containing the current workforce metrics.
        :param scorer_class: The ResilienceScorer class used to calculate the scores.
        """
        self.base_data = base_data.copy()
        self.scorer_class = scorer_class
        
        # Calculate baseline score
        self.baseline_scorer = self.scorer_class(self.base_data)
        self.baseline_results = self.baseline_scorer.evaluate()
        self.baseline_score = self.baseline_results['resilience_score']

    def simulate_scenario(self, headcount_change_pct: float = 0.0, target_overtime_reduction: float = 0.0, efficiency_improvement: float = 0.0) -> dict:
        """
        Simulates changes to the workforce and calculates a Projected Resilience Score.
        
        :param headcount_change_pct: Percentage change in headcount (e.g., 0.10 for +10%).
                                     Simulated by modifying the Scheduled_Hrs capacity.
        :param target_overtime_reduction: Percentage to reduce overtime by (e.g., 0.20 for 20% reduction).
                                          Simulated by reducing Overtime_Hrs.
        :param efficiency_improvement: Percentage improvement to Focus_Efficiency (e.g., 0.05 for 5% improvement).
        :return: A dictionary with projected score, delta, and a narrative summary.
        """
        sim_data = self.base_data.copy()
        
        # Apply Headcount Change
        # Increasing headcount increases available scheduled hours capacity
        if 'Scheduled_Hrs' in sim_data.columns:
            sim_data['Scheduled_Hrs'] = sim_data['Scheduled_Hrs'] * (1.0 + headcount_change_pct)
            
        # Apply Overtime Reduction
        # Target overtime reduction means reducing existing overtime hours
        if 'Overtime_Hrs' in sim_data.columns:
            sim_data['Overtime_Hrs'] = sim_data['Overtime_Hrs'] * (1.0 - target_overtime_reduction)
            # Ensure no negative overtime
            sim_data['Overtime_Hrs'] = np.maximum(sim_data['Overtime_Hrs'], 0.0)
            
        # Apply Efficiency Improvement
        if 'Focus_Efficiency' in sim_data.columns:
            # Assume improvement is relative (e.g., 5% better efficiency)
            sim_data['Focus_Efficiency'] = sim_data['Focus_Efficiency'] * (1.0 + efficiency_improvement)
            # Cap efficiency at 1.0 (100%)
            sim_data['Focus_Efficiency'] = np.minimum(sim_data['Focus_Efficiency'], 1.0)
            
        # Evaluate projected data
        projected_scorer = self.scorer_class(sim_data)
        projected_results = projected_scorer.evaluate()
        projected_score = projected_results['resilience_score']
        
        # Calculate Delta
        delta = round(projected_score - self.baseline_score, 2)
        
        # Generate Narrative
        narrative_parts = []
        if headcount_change_pct != 0:
            direction = "Increasing" if headcount_change_pct > 0 else "Decreasing"
            narrative_parts.append(f"{direction.lower()} headcount by {abs(headcount_change_pct)*100:.0f}%")
            
        if target_overtime_reduction != 0:
            narrative_parts.append(f"reducing overtime by {target_overtime_reduction*100:.0f}%")
            
        if efficiency_improvement != 0:
            narrative_parts.append(f"improving efficiency by {efficiency_improvement*100:.0f}%")
            
        if len(narrative_parts) > 1:
            action_desc = ", ".join(narrative_parts[:-1]) + ", and " + narrative_parts[-1]
        elif len(narrative_parts) == 1:
            action_desc = narrative_parts[0]
        else:
            action_desc = "making no changes"
            
        # Capitalize first letter of the sentence
        action_desc = action_desc.capitalize()
        
        verb = "improves" if delta > 0 else "reduces" if delta < 0 else "leaves"
        
        if delta == 0:
            narrative = f"{action_desc} {verb} the Resilience Score unchanged at {self.baseline_score}."
        else:
            narrative = f"{action_desc} {verb} the Resilience Score from {self.baseline_score} to {projected_score}."
            
        return {
            'baseline_score': self.baseline_score,
            'projected_score': projected_score,
            'score_delta': delta,
            'narrative_comparison': narrative,
            'projected_metrics': projected_results['metrics_dict']
        }

    def get_recommendations(self, projected_metrics: dict, top_negative_factor: str = None) -> list:
        """
        Evaluates the simulation results and returns a bulleted list of actionable recommendations.
        
        :param projected_metrics: The metrics dictionary returned by the simulation (e.g., scenario['projected_metrics']).
        :param top_negative_factor: Optional. The primary risk factor identified by the SHAP explainer.
        :return: A list of strings, each representing a bulleted recommendation.
        """
        recommendations = []
        
        # 1. Focus Efficiency Check
        if projected_metrics.get('Focus_Efficiency', 1.0) < 0.65:
            recommendations.append("- Prioritize Focus Efficiency training and AI-workflow optimization over new hiring to maximize resilience.")
            
        # 2. Primary Driver Check (Absenteeism)
        is_absenteeism_driver = False
        if top_negative_factor in ['Absenteeism_Rate', 'Absenteeism']:
            is_absenteeism_driver = True
        elif top_negative_factor is None:
            # Estimate the driver based on internal penalty weights if SHAP isn't provided
            abs_penalty = min(projected_metrics.get('Absenteeism_Rate', 0) / 0.20, 1.0) * 30
            ot_penalty = min(projected_metrics.get('Overtime_Dependency', 0) / 0.30, 1.0) * 30
            att_penalty = min(projected_metrics.get('Attrition_Rate', 0) / 0.10, 1.0) * 40
            
            if abs_penalty > ot_penalty and abs_penalty > att_penalty and abs_penalty > 0:
                is_absenteeism_driver = True
                
        if is_absenteeism_driver:
            recommendations.append("- Implement targeted Workplace Well-being Programs (WWPs) to mitigate the high marginal cost of lost productivity (Absence Multiplier).")
            
        # 3. Overtime Check
        if projected_metrics.get('Overtime_Dependency', 0.0) > 0.15:
            recommendations.append("- Strategic headcount expansion is required to prevent systemic burnout and long-term attrition risk.")
            
        return recommendations

    def calculate_recovery_velocity(self, active_recommendations: list, pandemic_mode_active: bool = False) -> float:
        """
        Calculates the 'Total Monthly Velocity' based on the active recommendations.
        
        Improvement Coefficients:
        - Focus Efficiency Training: +2.5 points/month
        - New Headcount (Onboarding): +1.5 points/month
        - Wellbeing Programs: +1.0 points/month
        
        :param active_recommendations: A list of active recommendations.
        :param pandemic_mode_active: If True, applies a 25% 'Friction Penalty'.
        :return: Total monthly velocity in points per month.
        """
        velocity = 0.0
        
        for rec in active_recommendations:
            if "Focus Efficiency training" in rec:
                velocity += 2.5
            elif "Strategic headcount expansion" in rec:
                velocity += 1.5
            elif "Workplace Well-being Programs" in rec:
                velocity += 1.0
                
        # Apply 25% 'Friction Penalty' if in Pandemic Mode
        if pandemic_mode_active:
            velocity *= 0.75
            
        return round(velocity, 2)

    def estimate_recovery_time(self, target_score: float, current_score: float, active_recommendations: list, pandemic_mode_active: bool = False) -> dict:
        """
        Estimates the number of months required to reach the target score from the current simulated score.
        
        :param target_score: The desired Resilience Score.
        :param current_score: The current (or projected) Resilience Score.
        :param active_recommendations: A list of active recommendations.
        :param pandemic_mode_active: If True, applies a 25% 'Friction Penalty'.
        :return: A dictionary with the estimated months and a narrative string.
        """
        if current_score >= target_score:
            return {
                'months': 0.0,
                'narrative': f"Current score ({current_score}) already meets or exceeds the target ({target_score})."
            }
            
        velocity = self.calculate_recovery_velocity(active_recommendations, pandemic_mode_active)
        
        if velocity <= 0.0:
            return {
                'months': float('inf'),
                'narrative': "Without active interventions, recovery to the target score cannot be estimated."
            }
            
        score_gap = target_score - current_score
        months_required = score_gap / velocity
        
        narrative = f"Estimated recovery time is {months_required:.1f} months to reach target score {target_score} at a velocity of {velocity} pts/month."
        if pandemic_mode_active:
            narrative += " (Includes a 25% friction penalty due to pandemic mode)."
            
        return {
            'months': round(months_required, 1),
            'narrative': narrative
        }
