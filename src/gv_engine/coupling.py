def coupling_drift(
    prev_value: float,
    curr_value: float,
    inflow: float,
    base_leak_rate: float,
    min_val: float,
    max_val: float,
) -> float:
    """
    Measures counterfactual deviation:
    How much did this constraint change vs how it *would have* changed independently.
    Returns value in [0, 1].
    """

    actual_delta = curr_value - prev_value

    expected_delta = inflow - base_leak_rate * (prev_value - min_val)

    denom = abs(expected_delta) + 1e-9
    drift = abs(actual_delta - expected_delta) / denom

    return min(max(drift, 0.0), 1.0)
