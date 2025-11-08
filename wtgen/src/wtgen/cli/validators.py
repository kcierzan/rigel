def validate_tilt_string(type_: object, eq_string: str | None) -> None:
    if not eq_string:
        return

    parts = eq_string.split(":")
    if len(parts) != 2:
        raise ValueError("Tilt string must be in format 'start_ration:gain_db'")

    start_ratio = float(parts[0])

    if not 0.0 <= start_ratio <= 1.0:
        raise ValueError("Start ration must be between 0.0 and 1.0")


def validate_eq_string(type_: object, eq_string: str | None) -> None:
    if not eq_string:
        return

    for band_str in eq_string.split(","):
        parts = band_str.strip().split(":")

        if len(parts) < 2 or len(parts) > 3:
            raise ValueError(f"Invalid EQ band format: {band_str}")


def validate_power_of_two_integer(type_: object, size: int) -> None:
    """Validate that size is a power of 2."""
    if size & (size - 1) != 0:
        raise ValueError("Size must be a power of 2")
