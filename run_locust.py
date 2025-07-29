#!/usr/bin/env python3
"""
Helper script to run Locust load testing for the NER API.

Usage:
    python run_locust.py

This will start Locust with the default configuration.
You can also run it manually with:
    locust -f locustfile.py --host=http://localhost:8000
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Check if full_data.csv exists
    data_path = Path("files/datasets/full_data.csv")
    if not data_path.exists():
        print(f"Error: {data_path} not found!")
        print("Please make sure the full_data.csv file exists in the files/datasets/ directory.")
        sys.exit(1)
    
    # Set environment variable for the data path
    os.environ["LOCUST_PAYLOAD_DATA_PATH"] = str(data_path.parent)
    
    print("Starting Locust load testing...")
    print(f"Data path: {data_path.parent}")
    print("Make sure your API is running on http://localhost:8000")
    print("Locust will be available at http://localhost:8089")
    print("\nPress Ctrl+C to stop the load test")
    
    try:
        # Run locust
        subprocess.run([
            "locust", 
            "-f", "locustfile.py",
            "--host=http://localhost:8000",
            "--web-host=0.0.0.0",
            "--web-port=8089"
        ])
    except KeyboardInterrupt:
        print("\nLoad testing stopped by user")
    except FileNotFoundError:
        print("Error: Locust not found!")
        print("Please install Locust first:")
        print("pip install locust")
        sys.exit(1)

if __name__ == "__main__":
    main() 