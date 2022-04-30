import pandas as pd

from special_time_filters import (create_ascending_number_filter,
                                  create_exact_hour_filter,
                                  create_is_palindromic_filter,
                                  create_repeating_doubles_filter,
                                  create_same_number_filter,
                                  create_three_pairs_filter,
                                  create_two_pairs_filter)


def generate_all_times():
    """
    This function defines all possible times, for legibility we will define each possible
    digit as its own column.
    """
    possible_digits = {
        'ten_hours': [i for i in range(3)],
        'hours': [i for i in range(10)],
        'ten_minutes': [i for i in range(6)],
        'minutes': [i for i in range(10)]
    }
    all_times = {
        'ten_hours': [],
        'hours': [],
        'ten_minutes': [],
        'minutes': []
    }

    for ten_hour in possible_digits['ten_hours']:
        for hour in possible_digits['hours']:
            if ten_hour == 2 and hour > 3:
                continue
            for ten_minute in possible_digits['ten_minutes']:
                for minute in possible_digits['minutes']:
                    all_times['ten_hours'].append(ten_hour)
                    all_times['hours'].append(hour)
                    all_times['ten_minutes'].append(ten_minute)
                    all_times['minutes'].append(minute)

    return pd.DataFrame(all_times)


def get_all_special_times(df=None):
    if df is None:
        df = generate_all_times()

    return df[
        (create_ascending_number_filter(df)) |
        (create_exact_hour_filter(df)) |
        (create_is_palindromic_filter(df)) |
        (create_repeating_doubles_filter(df)) |
        (create_same_number_filter(df)) |
        (create_three_pairs_filter(df)) |
        (create_two_pairs_filter(df))
    ]
