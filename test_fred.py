import os
from dotenv import load_dotenv
from intelligence_layer.fred_integration import FredIntegration

load_dotenv()

print("=== Testing FRED Integration ===")

api_key = os.environ.get('FRED_API_KEY')
if not api_key:
    print("Warning: FRED_API_KEY environment variable is not set or not detected by this process.")
else:
    print(f"FRED_API_KEY detected (starts with {api_key[:4]}...)")

try:
    fred_api = FredIntegration()
    unemployment = fred_api.get_current_unemployment_rate()
    if unemployment is not None:
        print(f"Success! Current US Unemployment Rate: {unemployment}%")
    else:
        print("Failed: No data returned.")
except Exception as e:
    print(f"Error during FRED execution: {e}")
