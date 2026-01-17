def coupling_drift(a: float, b: float, strength: float = 0.01) -> float:
    """
    Simple symmetric coupling drift.
    """
    return strength * (a - b)
