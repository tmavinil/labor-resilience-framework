import os
import sys
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

class FredIntegration:
    def __init__(self, api_key: str = None):
        """
        Initializes the FRED API integration.
        Requires an API key, either passed directly or set as the 'FRED_API_KEY' environment variable.
        """
        load_dotenv()
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        
        if not self.api_key:
            print("Error: A FRED API key is required but was not found.", file=sys.stderr)
            print("Please create a '.env' file in the root directory and add 'FRED_API_KEY=your_key_here'.", file=sys.stderr)
            self.fred = None
        else:
            self.fred = Fred(api_key=self.api_key)

    def get_us_unemployment_rate(self) -> pd.Series:
        """
        Fetches the historical and current US Unemployment Rate (Series ID: UNRATE).
        Returns a pandas Series with dates as the index.
        """
        if not self.fred:
            print("Warning: Skipping FRED fetch because the API key is missing. Returning empty series.")
            return pd.Series(dtype=float)
            
        try:
            print("Fetching US Unemployment Rate from FRED...")
            # 'UNRATE' is the FRED series ID for the US Unemployment Rate
            unrate_data = self.fred.get_series('UNRATE')
            return unrate_data
        except Exception as e:
            print(f"Error fetching data from FRED: {e}")
            return pd.Series(dtype=float)

    def get_current_unemployment_rate(self) -> float:
        """
        Fetches the most recent available US Unemployment Rate.
        """
        unrate_data = self.get_us_unemployment_rate()
        if not unrate_data.empty:
            # Get the last non-null value
            latest_rate = unrate_data.dropna().iloc[-1]
            return float(latest_rate)
        return None
