from gv_engine import Constraint, gv_score


def test_score_increases_with_utilization():
    low = Constraint("cpu", value=40, capacity=100)
    high = Constraint("cpu", value=90, capacity=100)
    assert gv_score([high]) > gv_score([low])


def test_over_capacity_penalty():
    ok = Constraint("cpu", value=90, capacity=100)
    bad = Constraint("cpu", value=140, capacity=100)
    assert gv_score([bad]) > gv_score([ok])
