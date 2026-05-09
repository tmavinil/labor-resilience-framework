import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_records=15000, output_path="examples/synthetic_labor_data.csv"):
    """
    Generate synthetic labor records for multiple industries.
    Correlates overtime, absenteeism, and attrition with a Macro Economic Stability index.
    """
    print(f"Generating {num_records} synthetic labor records...")
    
    np.random.seed(42)
    industries = ['Healthcare', 'Retail', 'Manufacturing']
    
    # Generate dates over the last 2 years
    start_date = datetime(2024, 1, 1)
    # Random days offset from start_date
    days_offset = np.random.randint(0, 730, num_records)
    dates = [start_date + timedelta(days=int(d)) for d in days_offset]
    
    # Create a Macro Economic Stability Index (0.0 to 1.0)
    # Use a sine wave to simulate economic cycles, plus some noise
    # Higher index = more stable economy
    macro_economic_stability = 0.5 + 0.3 * np.sin(days_offset / 100.0) + np.random.normal(0, 0.05, num_records)
    macro_economic_stability = np.clip(macro_economic_stability, 0, 1)
    
    # Generate basic fields
    industry_col = np.random.choice(industries, num_records)
    employee_id_col = np.random.randint(10000, 99999, num_records)
    
    # Base scheduled hours (mostly 40, some part-time)
    scheduled_hrs_col = np.random.choice([40, 35, 30, 20], num_records, p=[0.7, 0.15, 0.1, 0.05])
    
    # Calculate correlated metrics:
    # Lower stability -> higher overtime, higher absenteeism, higher attrition
    
    # 1. Overtime: Probability increases as stability decreases
    base_overtime_prob = 0.1
    overtime_prob = base_overtime_prob + (1 - macro_economic_stability) * 0.4
    has_overtime = np.random.rand(num_records) < overtime_prob
    overtime_hrs_col = np.zeros(num_records)
    # Random overtime hours between 1 and 12
    overtime_hrs_col[has_overtime] = np.random.uniform(1, 12, size=has_overtime.sum())
    
    # 2. Absenteeism: Probability increases as stability decreases
    base_absent_prob = 0.03
    absent_prob = base_absent_prob + (1 - macro_economic_stability) * 0.15
    is_absent_col = np.random.rand(num_records) < absent_prob
    
    # 3. Attrition (Terminated): Probability increases as stability decreases
    base_term_prob = 0.01
    term_prob = base_term_prob + (1 - macro_economic_stability) * 0.08
    terminated_col = np.random.rand(num_records) < term_prob
    
    # Calculate Actual Hours
    # If absent, actual hours are 0, else scheduled + overtime
    actual_hrs_col = np.where(is_absent_col, 0, scheduled_hrs_col) + overtime_hrs_col
    
    # Compile DataFrame
    df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in dates],
        'Industry': industry_col,
        'Employee_ID': employee_id_col,
        'Scheduled_Hrs': scheduled_hrs_col,
        'Actual_Hrs': actual_hrs_col.round(1),
        'Overtime_Hrs': overtime_hrs_col.round(1),
        'Is_Absent': is_absent_col.astype(int),
        'Terminated': terminated_col.astype(int),
        'Macro_Economic_Stability': macro_economic_stability.round(3)
    })
    
    # Sort by date for chronological consistency
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully generated and saved data to '{output_path}'.")
    
if __name__ == "__main__":
    # Generate exactly 15,000 records
    generate_synthetic_data(num_records=15000, output_path="examples/synthetic_labor_data.csv")
