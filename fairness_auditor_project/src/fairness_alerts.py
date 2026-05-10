def fairness_alert_system(dp, di, counterfactual_rate):

    alerts = []

    if dp > 0.10:
        alerts.append("High Demographic Parity Difference")

    if di < 0.80:
        alerts.append("Disparate Impact below fairness threshold")

    if counterfactual_rate > 0.10:
        alerts.append("High counterfactual bias rate")

    if len(alerts) == 0:
        print("Fairness status: ACCEPTABLE")

    else:
        print("\nFairness Alerts:")

        for alert in alerts:
            print("-", alert)

    return alerts