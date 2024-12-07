#!/usr/bin/env python3
"""
Script to fetch and print the location of a GitHub user using the GitHub API.

Usage:
    ./2-user_location.py <API_URL>

Parameters:
    API_URL: The full API URL of the user. Example: https://api.github.com/users/holbertonschool

Output:
    - Prints the location of the user if available.
    - If the user doesn’t exist, prints "Not found".
    - If the status code is 403 (rate limit exceeded), prints "Reset in X min" where X is the
      number of minutes until the rate limit resets.
    - Handles unexpected HTTP status codes or errors gracefully.

Notes:
    - The script ensures it does not execute when imported by using `if __name__ == '__main__':`.
"""

import sys  # For accessing command-line arguments
import requests  # For sending HTTP requests
import time  # For handling rate-limit reset calculations

if __name__ == '__main__':
    # Ensure the script is executed with the correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <API_URL>")
        sys.exit(1)

    # Get the API URL from the command-line argument
    api_url = sys.argv[1]

    try:
        # Send a GET request to the provided API URL
        response = requests.get(api_url)

        # Handle case where rate limit is exceeded
        if response.status_code == 403:
            # Extract the reset time from the headers, defaulting to the current time if unavailable
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time()))
            # Calculate the remaining time until reset in minutes
            reset_in_minutes = int((reset_time - time.time()) / 60)
            print(f"Reset in {reset_in_minutes} min")

        # Handle case where the user is not found
        elif response.status_code == 404:
            print("Not found")

        # Handle successful response (status code 200)
        elif response.status_code == 200:
            # Parse the JSON response
            user_data = response.json()
            # Extract the location field, with a fallback message if not available
            location = user_data.get('location', 'Location not available')
            print(location)

        # Handle unexpected status codes
        else:
            print(f"Unexpected status code: {response.status_code}")

    # Handle potential request exceptions
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
