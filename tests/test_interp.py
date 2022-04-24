from lumo.core.interp import Linear, Cos, PeriodCos, PeriodLinear, PeriodHalfCos


def test_continous():
    assert 0.5 == Linear.interp(5, start=0, end=1, left=0, right=10)
    assert 0.5 == Linear(start=0, end=1, left=0, right=10)(5)
    assert 0.5 == Cos.interp(5, start=0, end=1, left=0, right=10)
    assert 0.5 == Cos(start=0, end=1, left=0, right=10)(5)

    assert 0 == Cos(start=0, end=1, left=0, right=10, constant=True)(5)


def test_period():
    assert 0.5 == PeriodLinear.interp(15, start=0, end=1, left=0, period=10)
    assert 0.5 == PeriodHalfCos.interp(15, start=0, end=1, left=0, period=10)
