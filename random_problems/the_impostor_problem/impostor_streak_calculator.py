from itertools import product
from math import factorial

import numpy as np
from graphviz import Digraph
from utils import combine_state_and_masks, fix_state_order

mask_transition_probabilities_type = dict[tuple[int], float]


class ImpostorStreakCalculator:
    """Calculates the transition matrix and probabilities of achieving a certain impostor streak."""

    def __init__(
        self,
        number_of_impostors: int,
        number_of_players: int,
        streak_target: int,
        rounds_played: int,
    ) -> None:
        """The following parameters needed to initialise the calculator:
        - number_of_impostors = How many impostors are chosen each round.
        - number_of_players = How many players are playing the game.
        - streak_target = What minimum streak length you want to achieve.
        - rounds_played = How many rounds are being played in total.
        """
        self.number_of_impostors = number_of_impostors
        self.number_of_players = number_of_players
        self.streak_target = streak_target
        self.rounds_played = rounds_played
        self.transitions_probabilities = None

    def calculate_probability_of_streak(self) -> float:
        """
        First we need to generate the Markov matrix to quickly calculate the
        transition probabilities from any state to any other state after a given
        number of rounds.

        To get the final probability of a given streak we have to sum all the
        probabilities of going from the starting state to any state that contains
        the streak we are aiming for.

        We have to play at least one round for the calculation makes sense.
        """
        markov_chain_matrix, end_state_indices = self.create_markov_chain_matrix()

        probabilities_after_r_rounds = np.linalg.matrix_power(
            markov_chain_matrix, self.rounds_played
        )

        return probabilities_after_r_rounds[0][end_state_indices].sum()

    def create_markov_chain_matrix(self) -> tuple[np.ndarray, list[int]]:
        """
        Here we turn the transition probabilities into a Markov chain matrix.
        This allows us to raise the matrix to a given power to compute the
        probabilities an after an arbitrary amount of rounds in an efficient
        manner.

        We do lose readability when we convert to this matrix, but the only thing
        that we need to actually need to know is what columns we need to sum over
        to get the total probability at the end.
        """
        self.transitions_probabilities = self.create_state_transition_probabilities()

        state_indexes = {
            key: id
            for id, key in enumerate(sorted(self.transitions_probabilities.keys()))
        }
        transition_matrix = np.zeros((len(state_indexes), len(state_indexes)))

        for from_state in self.transitions_probabilities:
            for to_state in self.transitions_probabilities[from_state]:
                transition_matrix[
                    state_indexes[from_state], state_indexes[to_state]
                ] = self.transitions_probabilities[from_state][to_state]

        end_state_indices = [
            index for key, index in state_indexes.items() if self.streak_target in key
        ]

        return (transition_matrix, end_state_indices)

    def create_state_transition_probabilities(
        self,
    ) -> dict[tuple[int], dict[tuple[int], float]]:
        """
        Here we create a transition dict where we show the probability from each
        state to each state.

        Example:

        {
            (0,): {
                (1,): 1.0
            },
            (1,): {
                (1,): 0.9,
                (2,): 0.1
            },
            (2,): {
                (1,): 0.9,
                (3,): 0.1
            },
            (3,): {
                (3,): 1.0
            }
        }
        """
        mask_transition_probabilities = self.calculate_mask_transition_probabilities()
        return {
            state: self.add_transitions_from_state(state, mask_transition_probabilities)
            for state in self.generate_all_states()
        }

    def calculate_mask_transition_probabilities(
        self,
    ) -> mask_transition_probabilities_type:
        """
        This is the least intuitive part of the code. For brevity we will use the
        following abbreviations in the formulas below:
            - i = the number of impostors
            - p = the total number of players in a game
            - r = the total number of repeat impostors that are chosen next round

        To understand the logic better please note the following:
        1. The probability of a transition is uniquely dependent on the number of
        repeats in a mask. This is due to the fact that each mask with an
        equal number of repeats is equally likely to occur.
        2. The formula below can be split into 3 parts:
                - How many ways there are to chose an existing impostor again:
                (i!) / ((i - r)!)
                - How many ways are there to choose a new impostor:
                ((p - i)!) / ((p - 2i + r)!)
                - The sum of the previous two quantities:
                (p!) / ((p - i)!)
        """
        return {
            mask: (factorial(self.number_of_impostors))
            / (factorial(self.number_of_impostors - sum(mask)))
            * (factorial(self.number_of_players - self.number_of_impostors)) ** 2
            / (
                factorial(
                    self.number_of_players - 2 * self.number_of_impostors + sum(mask)
                )
            )
            / (factorial(self.number_of_players))
            for mask in self.generate_impostor_masks()
        }

    def generate_impostor_masks(self) -> list[tuple[int]]:
        """
        A mask indicates which of the previous impostors get chosen again in the
        next round (e.g. (1,0,1,1,0)). A 1 is also referred to as a repeat, a 0 is
        called a non-repeat.
        The masks must fulfill the following special condition:
        The number of non-repeats cannot be more than the number of players that
        were not an impostor in the last round. This is because for each non-repeat
        you need someone that was not the impostor last time.

        Example:
        With 3 players and two impostors these are the impostor masks:
        [(0, 1), (1, 0), (1, 1)]

        (0, 0) is not possible in this case as max one person was not an impostor
        last time. This means that it is not possible to find 2 people in the group
        that were not previously impostors.

        Additionally, the order of the masks does matter and are seen as different.
        This is different from how we count the states as unique.
        """
        return [
            mask
            for mask in list(product([0, 1], repeat=self.number_of_impostors))
            if self.number_of_impostors - sum(mask)
            <= self.number_of_players - self.number_of_impostors
        ]

    def add_transitions_from_state(
        self,
        state: tuple[int],
        mask_transition_probabilities: mask_transition_probabilities_type,
    ) -> dict[tuple[int], float]:
        """
        Here we transform the previously computed mask transition probabilities
        into the state transition probabilities. There are 2 notable differences
        between the two transition probabilities:
            - If we are in a state that contains the streak that we were trying to
            get to, then we do not leave that state.
            - Sometimes, multiple masks lead to the same next state. When this
            happens we sum up the probabilities of these masks. (e.g. If we are in
            the state (1, 1) then both the (1, 0) and the (0, 1) masks will lead
            to the new (2, 1) state).
        """
        if self.streak_target in state:
            return {state: 1.0}

        new_states_transition_probabilities = {}
        for mask in mask_transition_probabilities:
            new_state = fix_state_order(combine_state_and_masks(state, mask))
            new_states_transition_probabilities[new_state] = (
                new_states_transition_probabilities.get(new_state, 0)
                + mask_transition_probabilities[mask]
            )

        return new_states_transition_probabilities

    def generate_all_states(self) -> set[tuple[int]]:
        """
        Generates all possible states that the Markov chain could contain.
        Note that the states are determined only by their contents, not their
        order (e.g. (3, 1) is considered the same state as (1, 3)). For ease of
        reading they are sorted by descending order.

        Example:
        If the number of impostors is 2 and the max streak is 3 the
        possible states are:
        {(0, 0), (1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)}
        """
        return set(
            tuple(sorted(state, reverse=True))
            for state in product(
                [i for i in range(self.streak_target + 1)],
                repeat=self.number_of_impostors,
            )
            if not (0 in state and len(set(state)) > 1)
        )

    def create_markov_chain_graph(
        self, file_name: str = "markov_chain", save_locally: bool = False
    ) -> None | Digraph:
        graph = Digraph(format="jpeg")
        graph.attr(rankdir="LR", size="8,5")
        graph.attr("node", shape="circle")

        if self.transitions_probabilities is None:
            self.transitions_probabilities = (
                self.create_state_transition_probabilities()
            )

        nodelist = []

        for from_state in self.transitions_probabilities:
            for to_state in self.transitions_probabilities[from_state]:
                if (rate := self.transitions_probabilities[from_state][to_state]) > 0:
                    if from_state not in nodelist:
                        graph.node(str(from_state))
                        nodelist.append(to_state)
                    if to_state not in nodelist:
                        graph.node(str(to_state))
                        nodelist.append(to_state)

                    graph.edge(
                        str(from_state), str(to_state), label="{:.02f}".format(rate)
                    )

        if save_locally:
            graph.render(file_name, format="png", view=False)

        return graph
