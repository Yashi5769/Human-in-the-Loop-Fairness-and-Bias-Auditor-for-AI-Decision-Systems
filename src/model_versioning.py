import os
import pickle
import pandas as pd
from datetime import datetime

MODEL_DIR = "models"
LOG_FILE = "models/model_log.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

def save_model(model, dp):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"model_{timestamp}.pkl")

    # save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # log entry
    entry = {
        "timestamp": timestamp,
        "model_path": model_path,
        "dp": dp
    }

    df = pd.DataFrame([entry])

    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

    return model_path


def load_latest_model():
    if not os.path.exists(LOG_FILE):
        return None

    df = pd.read_csv(LOG_FILE)
    latest_path = df.iloc[-1]["model_path"]

    with open(latest_path, "rb") as f:
        return pickle.load(f)


def rollback_model():

    if not os.path.exists(LOG_FILE):
        return None

    df = pd.read_csv(LOG_FILE)

    if len(df) < 2:
        return None

    prev_model_path = df.iloc[-2]["model_path"]

    with open(prev_model_path, "rb") as f:
        return pickle.load(f)