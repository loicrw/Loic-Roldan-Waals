def create_same_number_filter(df):
    """
    Shows when all digits on the clock are the same.

    e.g.:
    - 11:11
    - 22:22
    """
    return ((df.ten_hours == df.hours) &
            (df.hours == df.ten_minutes) &
            (df.ten_minutes == df.minutes))


def create_is_palindromic_filter(df):
    """
    Shows when the time is palindromic.

    e.g.:
    - 12:21
    - 04:40
    """
    return ((df.ten_hours == df.minutes) &
            (df.hours == df.ten_minutes))


def create_two_pairs_filter(df):
    """
    Shows when the time has two pairs next to each other.

    e.g.:
    - 11:44
    - 22:55
    """
    return ((df.ten_hours == df.hours) &
            (df.ten_minutes == df.minutes))


def create_repeating_doubles_filter(df):
    """
    Shows when the hours and minutes are the same.

    e.g.:
    - 10:10
    - 06:06
    """
    return ((df.ten_hours == df.ten_minutes) &
            (df.hours == df.minutes))


def create_ascending_number_filter(df):
    """
    Shows when digits of the time are going up by one.

    e.g.:
    - 12:34
    - 23:45
    """
    return ((df.ten_hours == df.hours - 1) &
            (df.hours == df.ten_minutes - 1) &
            (df.ten_minutes == df.minutes - 1))


def create_three_pairs_filter(df):
    """
    Shows when three digits are the same.
    Note that the other digit should be 0. Why? ¯\_(ツ)_/¯

    e.g.:
    - 04:44
    - 11:01
    """
    # this matches the cases where the ten hours are zero and
    # the rest are all the same digit.
    ten_hour_zero = (
            (df.ten_hours == 0) &
            (df.hours == df.ten_minutes) &
            (df.ten_minutes == df.minutes)
    )

    # when the zero is in the hour position
    hour_zero = (
            (df.hours == 0) &
            (df.ten_hours == df.ten_minutes) &
            (df.ten_minutes == df.minutes)
    )

    # when the zero is in the ten minutes position
    ten_minute_zero = (
            (df.ten_minutes == 0) &
            (df.ten_hours == df.hours) &
            (df.hours == df.minutes)
    )

    # when the zero is in the minute position
    minute_zero = (
            (df.minutes == 0) &
            (df.ten_hours == df.hours) &
            (df.hours == df.ten_minutes)
    )
    return (
        ten_hour_zero |
        hour_zero |
        ten_minute_zero |
        minute_zero
    )


def create_exact_hour_filter(df):
    """
    Shows when it's exactly a given hour without any minutes.

    e.g.:
    - 04:00
    - 23:00
    """
    return ((df.ten_minutes == 0) &
            (df.minutes == 0))
