import pandas as pd
import os


def save_feedback(case_id, model_pred, human_decision):

    feedback = {
        "case_id": case_id,
        "model_prediction": model_pred,
        "human_decision": human_decision
    }

    df = pd.DataFrame([feedback])

    file_path = "/Users/yashi/Desktop/fairness_auditor_project/data/adult/human_feedback.csv"

    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, index=False)