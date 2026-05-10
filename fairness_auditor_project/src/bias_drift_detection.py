def detect_bias_drift(previous_dp, current_dp, threshold=0.05):
    """
    Detect drift in fairness metrics.

    Parameters
    ----------
    previous_dp : previous demographic parity value
    current_dp : current demographic parity value
    threshold : acceptable drift range

    Returns
    -------
    drift_flag : bool
    drift_value : float
    """

    drift_value = current_dp - previous_dp

    if abs(drift_value) > threshold:
        return True, drift_value

    return False, drift_value