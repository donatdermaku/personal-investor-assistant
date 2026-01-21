def assert_close(a: float, b: float, tol: float = 1e-6) -> None:
    if a is None or b is None:
        raise AssertionError("Values must be numeric.")
    if abs(a - b) > tol:
        raise AssertionError(f"{a} != {b} within tol={tol}")
