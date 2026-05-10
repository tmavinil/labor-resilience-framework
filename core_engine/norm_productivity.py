import pandas as pd
import numpy as np

def calculate_productivity_score(df: pd.DataFrame, hourly_cost: float = 35.0) -> pd.DataFrame:
    """
    Calculates the 'Productivity-Adjusted Overtime Score' to evaluate if overtime is yielding 
    proportionate output, or if the organization is suffering from diminishing returns.
    
    :param df: DataFrame containing ['Date', 'Overtime_Hrs', 'Actual_Hours' (or 'Actual_Hrs'), 'Productivity_Output']
    :param hourly_cost: Estimated cost per hour of overtime, used to calculate financial loss.
    :return: DataFrame with the new productivity scores, risk flags, and estimated losses.
    """
    # Create a copy to avoid SettingWithCopy warnings
    result_df = df.copy()
    
    # Handle column naming variations between systems
    actual_hours_col = 'Actual_Hours' if 'Actual_Hours' in result_df.columns else 'Actual_Hrs'
    if 'Overtime_Hrs' not in result_df.columns:
        # Default to 0 if missing, though it shouldn't be
        result_df['Overtime_Hrs'] = 0.0

    # Ensure data is sorted chronologically to evaluate trends over time
    if 'Date' in result_df.columns:
        result_df['Date'] = pd.to_datetime(result_df['Date'])
        result_df = result_df.sort_values(by='Date')
        
    # Prevent division by zero for Productivity Output
    # If Output is 0, we treat the ratio as NaN or a high penalty depending on logic
    safe_output = result_df['Productivity_Output'].replace(0, np.nan)
    
    # Calculate the core ratio: Overtime spent per unit of output
    result_df['OT_to_Output_Ratio'] = result_df['Overtime_Hrs'] / safe_output
    
    # Calculate the trend over time (expanding mean serves as a historical baseline)
    # If the current ratio is significantly higher than the historical average, efficiency is dropping
    result_df['Historical_OT_Ratio'] = result_df['OT_to_Output_Ratio'].expanding().mean().shift(1)
    
    # Fill NA in historical ratio with current ratio for the first row to establish a baseline
    result_df['Historical_OT_Ratio'] = result_df['Historical_OT_Ratio'].fillna(result_df['OT_to_Output_Ratio'])
    
    # Determine the increase in the ratio
    result_df['Ratio_Increase'] = result_df['OT_to_Output_Ratio'] - result_df['Historical_OT_Ratio']
    
    # Flag as High Productivity Risk if the ratio has worsened (increased) compared to historical baseline
    # We apply a small threshold to avoid flagging negligible micro-fluctuations
    threshold = 0.05 * result_df['Historical_OT_Ratio'] # 5% increase
    
    result_df['Productivity_Risk_Flag'] = np.where(
        result_df['Ratio_Increase'] > threshold,
        'High Productivity Risk',
        'Low/Normal Risk'
    )
    
    # Calculate Estimated Productivity Loss ($)
    # Loss = (Excess Overtime ratio) * Current Output * Hourly Cost
    # Essentially: "How many extra overtime hours did we spend compared to what we historically would have spent?"
    excess_overtime_hours = np.maximum(0, result_df['Ratio_Increase']) * result_df['Productivity_Output']
    result_df['Productivity_Loss_Dollars'] = (excess_overtime_hours * hourly_cost).round(2)
    
    # Calculate Productivity-Adjusted Overtime Score (0 to 100)
    # 100 is perfect efficiency (ratio is at or below historical baseline)
    # The score drops as the ratio gets worse compared to the baseline
    score_penalty = np.maximum(0, (result_df['Ratio_Increase'] / result_df['Historical_OT_Ratio'].replace(0, 0.01))) * 50
    result_df['Productivity_Adj_OT_Score'] = np.clip(100 - score_penalty, 0, 100).round(2)
    
    # Clean up intermediate calculation columns if desired, but we'll leave them for interpretability
    return result_df

def calculate_hybrid_productivity(actual_output: float, total_hours: float, focus_efficiency_score: float, overtime_hours: float = 0.0) -> dict:
    """
    Calculates the 'Real Output per Hour', applying a linear penalty to the yield
    if the focus_efficiency_score is below the 2026 national average of 0.60.
    Also reduces the yield by an additional 3% if overtime hours exceed 15% of total hours.
    
    :param actual_output: Total units of output produced.
    :param total_hours: Total hours worked (e.g. scheduled + overtime).
    :param focus_efficiency_score: Efficiency metric (0.0 to 1.0).
    :param overtime_hours: Total overtime hours worked.
    :return: A dictionary containing 'real_output_per_hour' and the adjusted 'hybrid_productivity_yield'.
    """
    if total_hours <= 0:
        return {
            'real_output_per_hour': 0.0,
            'hybrid_productivity_yield': 0.0
        }
        
    # Calculate 'Real Output per Hour'
    real_output_per_hour = actual_output / total_hours
    
    # Base yield starts as the real output per hour
    productivity_yield = real_output_per_hour
    
    # 2026 National Average
    national_avg_focus = 0.60
    
    # Apply linear penalty if focus_efficiency_score is below the 2026 national average
    # The penalty scales linearly down to 0 as the focus score drops to 0.
    if focus_efficiency_score < national_avg_focus:
        penalty_ratio = (national_avg_focus - focus_efficiency_score) / national_avg_focus
        productivity_yield -= (productivity_yield * penalty_ratio)
        
    # If overtime hours are > 15% of total hours, reduce by an additional 3%
    # to account for burnout-driven diminishing returns.
    if (overtime_hours / total_hours) > 0.15:
        productivity_yield *= 0.97
        
    return {
        'real_output_per_hour': round(real_output_per_hour, 4),
        'hybrid_productivity_yield': round(productivity_yield, 4)
    }
