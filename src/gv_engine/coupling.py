def coupling_drift(
    prev_value: float,
    curr_value: float,
    inflow: float,
    base_leak_rate: float,
    min_val: float,
    max_val: float,
) -> float:
    """
    Counterfactual deviation: how much did the system move
    differently than it would have independently?
    Returns value in [0, 1].
    """
    actual_delta = curr_value - prev_value
    expected_delta = inflow - base_leak_rate * (prev_value - min_val)

    if abs(expected_delta) < 1e-9:
        return 0.0

    drift = abs(actual_delta - expected_delta) / (max_val - min_val)
    return max(0.0, min(drift, 1.0))
