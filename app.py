import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to sys.path to allow importing project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from core_engine.resilience import ResilienceScorer
from optimization.simulator import WorkforceSimulator
from intelligence_layer.fred_integration import FredIntegration

load_dotenv()

st.set_page_config(page_title="Labor Resilience Framework", layout="wide")

st.title("Workforce Resilience Dashboard")

# Load baseline data
@st.cache_data
def load_data():
    csv_path = 'examples/synthetic_labor_data.csv'
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Data not found. Please run 'python examples/generate_data.py' first.")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Parameters")

headcount_change_pct = st.sidebar.slider(
    "Headcount Change (%)", 
    min_value=-20.0, 
    max_value=20.0, 
    value=0.0, 
    step=1.0
) / 100.0

baseline_ot_avg = df['Overtime_Hrs'].mean() if 'Overtime_Hrs' in df.columns else 0.0
ot_target_hours = st.sidebar.slider(
    "Overtime Target (Hours)", 
    min_value=0.0, 
    max_value=float(max(20.0, baseline_ot_avg * 2)), 
    value=float(round(baseline_ot_avg, 1)), 
    step=0.5
)

# Calculate reduction percentage for the simulator
if baseline_ot_avg > 0 and ot_target_hours < baseline_ot_avg:
    target_overtime_reduction = (baseline_ot_avg - ot_target_hours) / baseline_ot_avg
else:
    target_overtime_reduction = 0.0

baseline_efficiency = df['Focus_Efficiency'].mean() if 'Focus_Efficiency' in df.columns else 0.6
efficiency_goal = st.sidebar.slider(
    "Focus Efficiency Goal", 
    min_value=0.3, 
    max_value=1.0, 
    value=float(round(baseline_efficiency, 2)), 
    step=0.05
)

# Calculate efficiency improvement percentage
if baseline_efficiency > 0:
    efficiency_improvement = (efficiency_goal - baseline_efficiency) / baseline_efficiency
else:
    efficiency_improvement = 0.0

st.sidebar.divider()
st.sidebar.header("Stress Test")
pandemic_mode = st.sidebar.toggle("Pandemic Mode (Crisis Stress Test)", value=False)

# Prepare simulation data
sim_base_df = df.copy()

if pandemic_mode:
    st.sidebar.warning("⚠️ Pandemic Mode Active: Absenteeism spiked by 20%, Focus Efficiency reduced by 10%.")
    # Spike Absenteeism by 20%
    if 'Is_Absent' in sim_base_df.columns:
        # Increase the absenteeism rate by converting an additional 20% of the non-absent workforce to absent
        mask = sim_base_df['Is_Absent'] == 0
        indices_to_change = sim_base_df[mask].sample(frac=0.20, random_state=42).index
        sim_base_df.loc[indices_to_change, 'Is_Absent'] = 1
        
    # Reduce Focus Efficiency by 10%
    if 'Focus_Efficiency' in sim_base_df.columns:
        sim_base_df['Focus_Efficiency'] = sim_base_df['Focus_Efficiency'] * 0.90

# Initialize simulator and get baseline
# The baseline score is always based on the original data for true comparison
scorer_instance = ResilienceScorer(df)
baseline_score = scorer_instance.compute_score()

# We pass the (potentially pandemic-shocked) data to the simulator
simulator = WorkforceSimulator(base_data=sim_base_df, scorer_class=ResilienceScorer)

# Run Simulator
scenario = simulator.simulate_scenario(
    headcount_change_pct=headcount_change_pct,
    target_overtime_reduction=target_overtime_reduction,
    efficiency_improvement=efficiency_improvement
)

# Adjust scenario delta to be compared against the original baseline
projected_score = scenario['projected_score']
actual_delta = round(projected_score - baseline_score, 2)

# --- Main Page UI ---

st.header("Current vs. Projected Resilience Score")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Current Resilience Score (Baseline)", value=baseline_score)

with col2:
    st.metric(label="Projected Resilience Score", value=projected_score, delta=actual_delta)

st.info(f"**Impact Narrative:** {scenario['narrative_comparison']}")

# Show recommendations
st.header("Strategic Recommendations")
recommendations = simulator.get_recommendations(scenario['projected_metrics'])
if recommendations:
    for rec in recommendations:
        st.warning(rec)
else:
    st.success("- Current projected metrics are within healthy thresholds. No critical strategic interventions required.")

st.divider()

st.header("Recovery Path")
target_score = 80.0
velocity = simulator.calculate_recovery_velocity(recommendations, pandemic_mode_active=pandemic_mode)
recovery_est = simulator.estimate_recovery_time(target_score, projected_score, recommendations, pandemic_mode_active=pandemic_mode)

col_r1, col_r2 = st.columns(2)
with col_r1:
    velocity_display = f"+{velocity} points / month" if velocity > 0 else f"{velocity} points / month"
    st.metric(label="Recovery Velocity", value=velocity_display)

with col_r2:
    months_val = "N/A" if recovery_est['months'] == float('inf') else f"{recovery_est['months']} Months"
    if projected_score >= target_score:
        months_val = "Target Met"
    st.metric(label="Timeline to Baseline", value=months_val)

st.write("Recovery Trajectory")
if projected_score >= target_score:
    st.progress(1.0)
else:
    # Calculate progress toward 80, assuming 0 is the floor
    progress_pct = min(max(projected_score / target_score, 0.0), 1.0)
    st.progress(progress_pct)

st.caption("Recovery estimates are based on 2026 Labor Market Adaptability standards and current FRED economic multipliers.")

st.divider()

st.header("Historical Context")
plot_path = 'examples/resilience_trend_plot.png'
if os.path.exists(plot_path):
    st.image(plot_path, caption="Workforce Resilience Risk Factor Evolution (12-Month Trend)", use_container_width=True)
else:
    st.warning("Trend plot not found. Run 'python examples/run_trend_analysis.py' to generate it.")

st.divider()

st.header("Macro-Economic Stress Test")
if st.button("Run Economic Stress Test"):
    st.write("Fetching latest FRED data...")
    fred_api = FredIntegration()
    unemployment = fred_api.get_current_unemployment_rate()
    
    if unemployment is not None:
        st.success(f"Latest US Unemployment Rate from FRED: {unemployment}%")
        
        # Apply the new unemployment rate to the simulated baseline
        stress_df = df.copy()
        stress_df['National_Unemployment_Rate'] = unemployment
        
        stress_scorer = ResilienceScorer(stress_df)
        stress_score = stress_scorer.compute_score()
        
        delta_stress = round(stress_score - baseline_score, 2)
        
        st.metric(label="Stress-Tested Resilience Score (Current Workforce)", value=stress_score, delta=delta_stress)
        st.info("This stress test evaluates how the current workforce structure performs under the most recent macroeconomic conditions (real-time unemployment).")
    else:
        st.error("Could not fetch data from FRED. Please ensure your API key is valid.")
