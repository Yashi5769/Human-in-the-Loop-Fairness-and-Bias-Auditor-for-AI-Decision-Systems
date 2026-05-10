import pandas as pd

def generate_counterfactual(sample, feature_name):

    counterfactual = sample.copy()

    if counterfactual[feature_name] == 1:
        counterfactual[feature_name] = 0
    else:
        counterfactual[feature_name] = 1

    return counterfactual


def counterfactual_test(model, sample, feature_name):

    original_sample = sample.copy()
    counter_sample = generate_counterfactual(sample, feature_name)

    # ✅ FIX: preserve feature names
    original_df = pd.DataFrame([original_sample])[model.feature_names_in_]
    counter_df = pd.DataFrame([counter_sample])[model.feature_names_in_]

    original_prediction = model.predict(original_df)[0]
    counter_prediction = model.predict(counter_df)[0]

    return original_prediction, counter_prediction


def counterfactual_bias_rate(model, X_test, feature_name="sex_Male", samples=200):

    changes = 0

    for i in range(min(samples, len(X_test))):

        sample = X_test.iloc[i]

        sample_df = pd.DataFrame([sample])[model.feature_names_in_]

        counter = generate_counterfactual(sample, feature_name)
        counter_df = pd.DataFrame([counter])[model.feature_names_in_]

        original = model.predict(sample_df)[0]
        counter_pred = model.predict(counter_df)[0]

        if original != counter_pred:
            changes += 1

    return changes / samples