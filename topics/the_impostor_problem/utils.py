import numpy as np


def fix_state_order(state: tuple[int]) -> tuple[int]:
    return tuple(sorted(state, reverse=True))


def combine_state_and_masks(state: tuple[int], mask: tuple[int]) -> tuple[int]:
    return tuple(np.array(state) * np.array(mask) + 1)
