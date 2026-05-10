import pandas as pd
import numpy as np

def apply_zscore_normalization(org_data: pd.DataFrame, bls_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Z-score for a company's overtime hours relative to the industry
    mean and standard deviation fetched from the BLS, and returns a 'Resilience Penalty'.
    
    :param org_data: A DataFrame containing at least ['Industry', 'Overtime_Hrs'] for the organization.
    :param bls_stats: A DataFrame from BLS containing at least ['Industry', 'Overtime_Hrs'].
    :return: A DataFrame with the added columns 'Industry_Mean', 'Industry_Std', 'Z_Score', and 'Resilience_Penalty'.
    """
    # Group BLS data by industry to get mean and standard deviation
    # In case BLS data does not have enough points for std, we'll handle it.
    industry_metrics = bls_stats.groupby('Industry')['Overtime_Hrs'].agg(['mean', 'std']).reset_index()
    industry_metrics.rename(columns={'mean': 'Industry_Mean', 'std': 'Industry_Std'}, inplace=True)
    
    # Fill NaN std with a small default value if there's only one data point for the industry in BLS
    industry_metrics['Industry_Std'] = industry_metrics['Industry_Std'].fillna(0.1)
    
    # Ensure no zero standard deviation to avoid division by zero later
    industry_metrics['Industry_Std'] = industry_metrics['Industry_Std'].replace(0, 0.1)

    # Merge BLS stats into org data
    result_df = pd.merge(org_data, industry_metrics, on='Industry', how='left')
    
    # If there are industries in org_data not present in bls_stats, fill with overall fallback stats
    overall_mean = bls_stats['Overtime_Hrs'].mean() if not bls_stats.empty else 0
    overall_std = bls_stats['Overtime_Hrs'].std() if not bls_stats.empty else 1.0
    if pd.isna(overall_std) or overall_std == 0:
        overall_std = 1.0
        
    result_df['Industry_Mean'] = result_df['Industry_Mean'].fillna(overall_mean)
    result_df['Industry_Std'] = result_df['Industry_Std'].fillna(overall_std)
    
    # Calculate Z-score
    # Z = (X - mu) / sigma
    result_df['Z_Score'] = (result_df['Overtime_Hrs'] - result_df['Industry_Mean']) / result_df['Industry_Std']
    
    # Calculate Resilience Penalty
    # Starts accumulating only after Z-score > 1.5
    # For example, let's say penalty scales linearly beyond 1.5, multiplied by some base factor (e.g., 10 points per standard deviation)
    # Formula: max(0, Z_score - 1.5) * 10
    
    def calculate_penalty(z):
        if pd.isna(z):
            return 0.0
        if z > 1.5:
            # Scale penalty linearly for every standard deviation above 1.5
            return round((z - 1.5) * 10.0, 2)
        return 0.0
        
    result_df['Resilience_Penalty'] = result_df['Z_Score'].apply(calculate_penalty)
    
    return result_df
