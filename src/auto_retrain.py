from src.group_bias_mitigation import retrain_with_weights

def auto_retrain(df):

    model, X_test, y_test, preds = retrain_with_weights(df)

    return model, X_test, y_test, preds