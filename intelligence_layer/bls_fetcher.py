import os
import requests
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

class BLSFetcher:
    def __init__(self, api_key: str = None):
        """
        Initializes the BLS API fetcher.
        :param api_key: Optional BLS API Key. If not provided, it will check the 'BLS_API_KEY' environment variable.
                        Without an API key, BLS allows limited unauthenticated requests.
        """
        load_dotenv()
        self.api_key = api_key or os.environ.get('BLS_API_KEY')
        
        if not self.api_key:
            print("Notice: 'BLS_API_KEY' not found. Using unauthenticated mode.")
            print("Warning: Limits apply (50 series/request, 10 years of data, 25 queries/day).")            
           
        self.base_url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

    def fetch_overtime_data(self) -> pd.DataFrame:
        """
        Fetches Average Weekly Overtime hours for Manufacturing and Healthcare from BLS API v2.0.
        Returns a Pandas DataFrame with data for the last 12 months available.
        """
        # Map Series IDs to industry names
        # CEU3000000039: Manufacturing - Average weekly overtime hours of all employees
        # CEU6562000039: Health care and social assistance - Average weekly overtime hours of all employees
        series_dict = {
            'CEU3000000039': 'Manufacturing',
            'CEU6562000039': 'Healthcare'
        }
        
        # Calculate start and end years (BLS requires passing years for ranges)
        current_year = datetime.now().year
        start_year = current_year - 1
        
        headers = {'Content-type': 'application/json'}
        payload = {
            "seriesid": list(series_dict.keys()),
            "startyear": str(start_year),
            "endyear": str(current_year)
        }
        
        if self.api_key:
            payload["registrationkey"] = self.api_key
            
        print("Fetching Overtime data from BLS API...")
        response = requests.post(self.base_url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()
        
        json_data = response.json()
        
        if json_data.get('status') != 'REQUEST_SUCCEEDED':
            # It might partially succeed or fail, print any messages
            for message in json_data.get('message', []):
                print(f"BLS API Message: {message}")
            
        records = []
        # Parse the JSON response
        for series in json_data.get('Results', {}).get('series', []):
            series_id = series['seriesID']
            industry = series_dict.get(series_id, series_id)
            
            for item in series.get('data', []):
                year = item['year']
                period = item['period']
                value = item['value']
                
                # 'period' looks like 'M01', 'M02', ..., 'M12' (ignoring M13 for annual average)
                if period.startswith('M') and period != 'M13':
                    month = period[1:]
                    date_str = f"{year}-{month}-01"
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    records.append({
                        'Date': date_obj,
                        'Industry': industry,
                        'Overtime_Hrs': float(value)
                    })
                    
        if not records:
            print("No data found for the specified period.")
            return pd.DataFrame()
            
        df = pd.DataFrame(records)
        
        # Sort chronologically by industry and date
        df = df.sort_values(by=['Industry', 'Date'], ascending=[True, False])
        
        # Filter strictly to the last 12 months based on the most recent data point overall
        max_date = df['Date'].max()
        if pd.notnull(max_date):
            # 11 months subtraction to keep 12 distinct months inclusive
            cutoff_date = max_date - pd.DateOffset(months=11)
            df = df[df['Date'] >= cutoff_date].reset_index(drop=True)
            
        # Convert datetime back to ISO string for cleaner output/display
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        return df

if __name__ == "__main__":
    fetcher = BLSFetcher()
    try:
        overtime_df = fetcher.fetch_overtime_data()
        if not overtime_df.empty:
            print(f"\nSuccessfully fetched {len(overtime_df)} records (Last 12 Months):")
            print(overtime_df.to_string(index=False))
    except Exception as e:
        print(f"Failed to fetch BLS data: {e}")
