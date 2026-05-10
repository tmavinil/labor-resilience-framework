import pandas as pd
import os
import sys

# Add the parent directory to sys.path to allow importing project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interpretability.explainer import ResilienceExplainer
from core_engine.resilience import ResilienceScorer

def run_analysis():
    print("Loading synthetic dataset...")
    # Load the synthetic dataset we created in examples/generate_data.py
    df = pd.read_csv('examples/synthetic_labor_data.csv')
    
    # Sort the data by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Group by Month to create a time-series of metrics
    df['Month'] = df['Date'].dt.to_period('M')
    
    time_series_records = []
    
    print("Aggregating metrics by month to build time-series...")
    for period, group in df.groupby('Month'):
        # We can use the ResilienceScorer to consistently format our metrics for the explainer
        scorer = ResilienceScorer(group)
        metrics_df = scorer.get_shap_dataframe()
        time_series_records.append(metrics_df)
        
    # Combine the monthly snapshots into a single chronological DataFrame
    time_series_df = pd.concat(time_series_records, ignore_index=True)
    
    # Establish a baseline background dataset for SHAP (e.g., an 'ideal' or 'national average' state)
    # This provides the foundation for the SHAP Explainer to compare against
    background_baseline = pd.DataFrame([{
        'Absenteeism_Rate': 0.03, 
        'Overtime_Dependency': 0.05, 
        'Attrition_Rate': 0.01, 
        'Focus_Efficiency': 0.80, 
        'National_Unemployment_Rate': 4.0
    }])
    
    print("Initializing ResilienceExplainer...")
    explainer = ResilienceExplainer(background_baseline)
    
    print("Running Driver Drift analysis...")
    # --------------------------------------------------------------------------
    # NATIONAL LABOR RESILIENCE & DRIVER DRIFT
    # 
    # This analysis identifies 'Driver Drift' — determining if a specific workforce 
    # risk (e.g., Focus Efficiency or Overtime Dependency) is becoming a larger 
    # and more systemic contributor to workforce instability over time. 
    # By plotting these internal shifts against macroeconomic indicators, we can 
    # evaluate National Labor Resilience, forecasting shortages before they hit 
    # critical failure thresholds.
    # --------------------------------------------------------------------------
    trend_results = explainer.explain_trends(time_series_df)
    
    print("\n=== Analysis Results ===")
    print(trend_results.get('narrative_summary', 'No summary available.'))
    
    print("\nGenerating trend evolution visualization...")
    if 'shap_values' in trend_results:
        explainer.plot_trend_evolution(
            shap_values=trend_results['shap_values'],
            feature_names=explainer.feature_columns
        )

if __name__ == "__main__":
    run_analysis()
