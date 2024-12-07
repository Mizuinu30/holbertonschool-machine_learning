#!/usr/bin/env python3
import sys
import requests
import time

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <API_URL>")
        sys.exit(1)

    api_url = sys.argv[1]

    try:
        response = requests.get(api_url)
        if response.status_code == 403:
            # Handle rate limit
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time()))
            reset_in_minutes = int((reset_time - time.time()) / 60)
            print(f"Reset in {reset_in_minutes} min")
        elif response.status_code == 404:
            print("Not found")
        elif response.status_code == 200:
            user_data = response.json()
            location = user_data.get('location', 'Location not available')
            print(location)
        else:
            print(f"Unexpected status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
