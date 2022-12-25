# The Impostor Problem
## Introduction
This problem owes its name the the game "Among Us". In the game a minority of impostors try to sabotage and kill the other players. The conditions for winning the game are not really what we are going to focus on. The one thing we care about is what the chances are of being an impostor. More specifically we want to know **what the chances are of being an impostor multiple times in a row**. To calculate this we are going to make some assumptions:

- The impostor is selected randomly from all the players.
- Whether you were the impostor in a previous round has no impact on being chosen in subsequent rounds.
- The players do not change between rounds.

If you want to jump straight to the interactive calculator, click [here](https://the-impostor-problem.streamlit.app/)

## Generalized Problem Statement 
Usually a game of *Among Us* is played with 10 players and 2 impostors. However, we want to be able to answer the question for any arbitrary number of:

- Players playing the game.
- Impostors chosen each round.
- Rounds played.
- Streak achieved (i.e. if a player was the impostor 7 times in a row, the achieved streak is 7).

## Formalized Version
If we want to translate this problem to a more formalized version we would ask:

What is the probability $p$ that we get a streak of $s$ for any number, when we throw $d$ dice that are $n$-sided and play for $r$ rounds. 

An important thing to note here is that when we use dice we cannot throw the same number twice. So in the case that we get doubles (e.g. we throw 2 dice and get the numbers 5 and 5), we simply ignore the outcome and throw one or both dice again until we have two different numbers). The order that the dice thrown is irrelevant. Each round a number is either chosen or isn't.

We will once again list all the variables that we are considering, as well as their analogue in the original problem.

- $p$ - The probability that we are trying to find.
- $s$ - The streak we are trying to find. A streak is when a number/impostor is chosen in multiple consecutive rounds.
- $d$ - The number of dice we throw. This translates to the number of impostors that we choose. 
- $n$ - The number of possible choices for each die thrown. This translates to the number of players in a game.
- $r$ - How many rounds are played in total.

## Methodology
It is currently unknown how we can solve this problem. We have not yet confirmed a method that yields the correct result. There are however two methods that we identified as promising:

- Markov Chains
- Simulations

Markov sound like the most accurate method, but are conceptually harder as we need to dynamically generate the chain. Simulations however, are quite computationally intensive and are thus not suitable if we want a quick and lightweight calculation.

### Markov Chains
For the Markov Chain approach we simply keep track of the current streaks with states. The transitions between states indicate changes in current streaks. Here is an example of a simple tree (1 impostor, 10 players, max streak of 3):

![](Images/Simple%20chain.png)

They can however quickly get out of control (2 impostors, 10 players, max streak of 4):

![](Images/Complex%20chain.png)

#### Generating all possible states
TODO
#### Calculating the Transition Probabilities
TODO
#### Impostor Invariance
TODO
#### Impossible Transitions

## Conclusions
TODO