import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Function to fetch news for Apple (AAPL)
def fetch_apple_news(start_date="2024-01-11", end_date="2024-12-31", rate_limit=30):
    all_news = []
    
    # Format dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current_date = start
    request_count = 0  # Track the number of requests

    while current_date <= end:
        # Fetch news data for the current date
        params = {
            "apiKey": POLYGON_API_KEY,
            "ticker": "AAPL",  # Specify the ticker symbol for Apple
            "published_utc": current_date.strftime("%Y-%m-%d"),  # Date for news
            "limit": 100,  # Limit the number of news items per request
            "order": "desc",  # Order by most recent
            "sort": "published_utc"  # Sort by published date
        }
        
        print(f"Fetching news for {current_date.strftime('%Y-%m-%d')}...")
        
        response = requests.get(BASE_URL, params=params)
        
        # Check for successful response
        if response.status_code == 200:
            try:
                data = response.json()
                
                # Check if we got valid data and store it
                if "results" in data:
                    all_news.extend(data["results"])
                    print(f"Found {len(data['results'])} Apple-related news items for {current_date.strftime('%Y-%m-%d')}.")
                else:
                    print(f"No news found for {current_date.strftime('%Y-%m-%d')}.")
            except requests.exceptions.JSONDecodeError as e:
                print(f"Error decoding JSON for {current_date.strftime('%Y-%m-%d')}: {e}")
        else:
            print(f"Failed to fetch news for {current_date.strftime('%Y-%m-%d')} - Status code: {response.status_code}")
        
        # Increment request count
        request_count += 1

        # If 5 requests have been made, add a 40-second delay
        if request_count == 5:
            print(f"Reached 5 requests. Pausing for 40 seconds...")
            time.sleep(40)  # 40 seconds gap after every 5 requests
            request_count = 0  # Reset the counter after the pause
        
        # Respect rate limit by adding a delay between each request (5 requests per second)
        else:
            time.sleep(1 / rate_limit)  # Introduce delay to maintain rate limit
        
        # Move to the next day
        current_date += timedelta(days=1)
    
    # Return all collected news in a DataFrame
    return pd.DataFrame(all_news)

# Fetch and save Apple news
apple_news_df = fetch_apple_news(rate_limit=5)  # Adjust rate limit as per your API plan
if not apple_news_df.empty:
    apple_news_df.to_csv("apple_news_2.csv", index=False)
    print("CSV file saved successfully with Apple-related news for 2024.")
else:
    print("No news collected for Apple.")
