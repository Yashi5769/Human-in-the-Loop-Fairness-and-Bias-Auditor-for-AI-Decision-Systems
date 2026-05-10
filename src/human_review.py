import pandas as pd
import os
from datetime import datetime


def log_human_feedback(case_index, model_prediction, human_decision):

    # ✅ Create feedback dictionary (with timestamp included)
    feedback = {
        "case": case_index,
        "model_prediction": int(model_prediction),
        "human_decision": human_decision,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    feedback_df = pd.DataFrame([feedback])

    # ✅ Project-relative path (portable)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")

    # ✅ Ensure directory exists
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, "human_feedback.csv")

    # ✅ Check if file exists
    file_exists = os.path.isfile(file_path)

    # ✅ Append safely
    feedback_df.to_csv(
        file_path,
        mode="a",
        header=not file_exists,
        index=False
    )

    return True