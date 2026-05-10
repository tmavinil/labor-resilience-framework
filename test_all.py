import pandas as pd
import os
import sys

print("=== Starting Tests ===")
print("Loading synthetic data...")
try:
    df = pd.read_csv('examples/synthetic_labor_data.csv')
    print(f"Loaded {len(df)} records.")
except Exception as e:
    print(f"Failed to load synthetic data: {e}")
    sys.exit(1)

print("\n--- 1. Testing ResilienceScorer ---")
try:
    from core_engine.resilience import ResilienceScorer
    scorer = ResilienceScorer(df)
    results = scorer.evaluate()
    print("Resilience Score Results:")
    print({'resilience_score': results['resilience_score'], 'risk_category': results['risk_category']})
    metrics_df = results['metrics_df']
except Exception as e:
    print(f"Failed: {e}")

print("\n--- 2. Testing ResilienceExplainer (SHAP) ---")
try:
    from interpretability.explainer import ResilienceExplainer
    # Use proper baseline columns
    background_df = pd.DataFrame([
        {'Absenteeism_Rate': 0.05, 'Overtime_Dependency': 0.10, 'Attrition_Rate': 0.02, 'Focus_Efficiency': 0.70, 'National_Unemployment_Rate': 4.0},
        {'Absenteeism_Rate': 0.10, 'Overtime_Dependency': 0.15, 'Attrition_Rate': 0.05, 'Focus_Efficiency': 0.65, 'National_Unemployment_Rate': 4.5},
        {'Absenteeism_Rate': 0.20, 'Overtime_Dependency': 0.30, 'Attrition_Rate': 0.10, 'Focus_Efficiency': 0.50, 'National_Unemployment_Rate': 5.0},
    ])
    explainer = ResilienceExplainer(background_df)
    explanation = explainer.explain_score(metrics_df)
    print("Explanation Results:")
    print(explanation)
except Exception as e:
    print(f"Failed: {e}")

print("\n--- 3. Testing BLSFetcher ---")
try:
    from intelligence_layer.bls_fetcher import BLSFetcher
    bls_fetcher = BLSFetcher()
    bls_data = bls_fetcher.fetch_overtime_data()
    if not bls_data.empty:
        print(f"Fetched BLS data (Head):")
        print(bls_data.head(2))
    else:
        print("BLS data fetch succeeded but returned empty.")
except Exception as e:
    print(f"Failed: {e}")

print("\n--- 4. Testing FredIntegration ---")
try:
    from intelligence_layer.fred_integration import FredIntegration
    # We will test with a mock key, expect graceful failure/empty return
    fred_api = FredIntegration(api_key="mock_key_for_testing")
    unemployment = fred_api.get_current_unemployment_rate()
    print(f"Unemployment Rate: {unemployment}")
except Exception as e:
    print(f"Failed: {e}")

print("\n--- 5. Testing Z-Score Normalization ---")
try:
    from core_engine.norm_zscore import apply_zscore_normalization
    # Create mock BLS stats
    mock_bls = pd.DataFrame({
        'Industry': ['Healthcare', 'Retail', 'Manufacturing'],
        'Overtime_Hrs': [5.0, 3.0, 6.0]
    })
    zscore_df = apply_zscore_normalization(df.head(10), mock_bls)
    print(f"Z-Score Norm Output (Head):")
    print(zscore_df[['Industry', 'Overtime_Hrs', 'Industry_Mean', 'Industry_Std', 'Z_Score', 'Resilience_Penalty']].head(3))
except Exception as e:
    print(f"Failed: {e}")

print("\n--- 6. Testing Productivity Score ---")
try:
    from core_engine.norm_productivity import calculate_productivity_score, calculate_hybrid_productivity
    # Mock Productivity_Output since we only generate Focus_Efficiency
    test_df = df.head(50).copy()
    test_df['Productivity_Output'] = test_df['Actual_Hrs'] * test_df['Focus_Efficiency'] * 2
    prod_df = calculate_productivity_score(test_df)
    print(f"Productivity Adj Output (Head):")
    print(prod_df[['OT_to_Output_Ratio', 'Productivity_Risk_Flag', 'Productivity_Adj_OT_Score']].head(3))
    
    print("\nTesting calculate_hybrid_productivity...")
    hybrid_res = calculate_hybrid_productivity(
        actual_output=100.0, 
        total_hours=50.0, 
        focus_efficiency_score=0.55, 
        overtime_hours=10.0
    )
    print(f"Hybrid Productivity Result: {hybrid_res}")
except Exception as e:
    print(f"Failed: {e}")

print("\n=== Tests Completed ===")
