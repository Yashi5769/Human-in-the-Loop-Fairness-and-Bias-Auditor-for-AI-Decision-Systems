import pandas as pd
import os
from datetime import datetime

LOG_FILE = "data/fairness_log.csv"


def log_fairness(dp, di):

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dp": dp,
        "di": di
    }

    df = pd.DataFrame([log_entry])

    os.makedirs("data", exist_ok=True)

    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)


def check_drift(threshold=0.05):

    if not os.path.exists(LOG_FILE):
        return False, 0

    df = pd.read_csv(LOG_FILE)

    if len(df) < 2:
        return False, 0

    latest = df.iloc[-1]["dp"]
    previous = df.iloc[-2]["dp"]

    drift = abs(latest - previous)

    return drift > threshold, drift