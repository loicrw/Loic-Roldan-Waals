import streamlit as st
from graphviz import Digraph
from impostor_streak_calculator import ImpostorStreakCalculator

st.markdown(
    """
    # The Impostor Streak Calculator
    This calculator allows you to check what the chance is that any player has 
    an impostor streak of a certain length. Fill in the parameters below and 
    find out!
    """
)
number_of_impostors = st.slider(
    "How many impostors are there?", min_value=1, max_value=3
)
number_of_players = st.slider("How many people are playing?", min_value=4, max_value=15)
streak_target = st.slider(
    "What streak do you need to reach?", min_value=1, max_value=15
)
rounds_played = st.slider(
    "How many rounds are playing?", min_value=streak_target + 1, max_value=50
)

calculator = ImpostorStreakCalculator(
    number_of_impostors=number_of_impostors,
    number_of_players=number_of_players,
    streak_target=streak_target,
    rounds_played=rounds_played,
)

st.markdown("## Answer")
st.write(
    f"The probability of achieving this streak is {calculator.calculate_probability_of_streak():.5f}"
)

show_markov_chain = st.button(
    "Click here to visualise the Markov chain of streak states"
)
if show_markov_chain:
    graph = calculator.create_markov_chain_graph()
    if isinstance(graph, Digraph):
        st.graphviz_chart(graph)
