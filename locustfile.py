"""
Please install [Locust](https://docs.locust.io/en/stable) before running this.

Also, make sure you have your `full_data.csv` in the same directory as where you're running Locust,
otherwise please specify the path to the `full_data.csv` file via the LOCUST_PAYLOAD_DATA_PATH environment variable.
"""

from locust import HttpUser, task, between
import pandas as pd
import pathlib
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

payload_path = pathlib.Path(os.environ.get("LOCUST_PAYLOAD_DATA_PATH", "."))

print(f"Loading dataframe from {payload_path}")
df_payload = pd.read_csv(payload_path / "full_data.csv")
print(f"Loaded dataframe with shape {df_payload.shape} from {payload_path}")

class MLServiceLoadTest(HttpUser):
    wait_time = between(1, 5) # seconds

    @task
    def is_it_alive(self):
        print(f"Sending request to /healthz")
        self.client.get("/healthz")

    @task(10)
    def view_items(self):
        print(f"Sending request to /predict/single")
        self.client.post("/predict/single", json={"text": df_payload.sample(1).iloc[0].text})