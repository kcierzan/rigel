from typing import NoReturn

EPSILON = 1e-12


def assert_exhaustiveness(x: NoReturn) -> NoReturn:
    """Provide an assertion at type-check time that this function is never called."""
    raise AssertionError(f"Invalid value: {x!r}")
