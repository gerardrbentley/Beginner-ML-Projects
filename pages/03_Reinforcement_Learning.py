import itertools
import random
from typing import Optional

import gym
import numpy as np
import pandas as pd
import streamlit as st

st.subheader("(RL) Reinforcement Learning 🤖", "rl")

st.write(
    """'Q-Learning' policy optimization by simulating rounds of BlackJack.
The 'Q-Learning' algorithm will learn to make the best 'Action' choice for a given 'State Observation' in a Reinforcement Learning 'Environment' (see [wikipedia](https://en.wikipedia.org/wiki/Q-learning) for more).

Other awesome app features:
- Multiple Views
- Query Arguments in URL

Q roughly stands for "Quality".
It's the mathematical function we aim to learn: the function that predicts a precise 'Reward' based on a combination of 'State' and 'Action.'

Powered by [OpenAI Gym](https://www.gymlibrary.ml/) and [Streamlit](https://docs.streamlit.io/).
Built with ❤️ by [Gar's Bar](https://tech.gerardbentley.com/)
"""
)

with st.expander("Read More"):
    st.write(
        """
### In this example:

- 'State': Whatever cards are in the player's hand and the dealer's faceup card
- 'Action': Either "Hit" to get another card or "Stand" to give up
- 'Reward': Positive values if the player wins. Negative if they lose.

### BlackJack Strategies:

BlackJack involves the player collecting cards to sum closer to 21 than the dealer without going over.
The player starts with 2 cards then can choose to get another card (until they "bust" over 21) or stick with the current sum.

- Random Strategy: Pick "Hit" or "Stand" by metaphorically flipping a coin. Doesn't matter what current player hand is
- Stand at 17 Strategy: Pick "Hit" if the player sum is less than 17. Pick "Stand" if it is 17 or higher.
- Q-Learning Strategy: Observe the player sum, dealer hand, and whether player has an Ace, then decide whether "Hit" or "Stand" is better in that situation.

### Q-Learning Training:

We modify the environment in 2 ways to benefit the training.

- Limit Bust Sums: Reduce the number of state observations by limiting the bust sum to 22
- Incremental Rewards: The environment gives no reward for a "Hit" that doesn't bust, but we can use the following heuristics:
    - "Hit" when the player sum is less than 12 is always safe and should get a positive reward
    - "Stand" when the player sum is less than 12 wastes the opportunity to safely "Hit" and should receive a negative reward
    - "Hit" when the player sum is greater than 16 is risky and should receive a negative reward
    - "Stand" when the player sum is greater than 16 is always safe and should get a positive reward

"""
    )


random_play = "Random BlackJack Gameplay"
human_play = "Human BlackJack Gameplay"
trained_play = "Q-Learning BlackJack Gameplay"
query_params = st.experimental_get_query_params()
(default_view,) = query_params.get("view", [random_play.replace(" ", "+")])

if "view" not in st.session_state:
    st.session_state.view = default_view.replace("+", " ")
view = st.sidebar.radio(
    "Which Strategy to Demo",
    [random_play, human_play, trained_play],
    key="view",
)
st.experimental_set_query_params(view=view)
game_seed = int(st.sidebar.number_input("Game Seed", 0, 100, 47))
number_of_games = int(st.sidebar.number_input("Number of Games", 1, 100000, 20000))
if view == trained_play:
    alpha = st.sidebar.number_input("Learning Rate (alpha)", 0.001, 1.0, 0.01, 0.05)
    gamma = st.sidebar.number_input("Discount Factor (gamma)", 0.0, 1.0, 0.5, 0.05)
    epsilon = st.sidebar.number_input("Exploration Factor (epsilon)", 0.0, 1.0, 0.1)
    training_steps = int(
        st.sidebar.number_input("Training Episodes", 1, 1000000, 20000, 1000)
    )
    # training_steps = int(st.sidebar.number_input("Training Episodes", 10, 1000000, 20000, 1000))
environment = gym.make("Blackjack-v1")
environment.action_space.seed(game_seed)
environment.reset(seed=game_seed)
random.seed(game_seed)

combinations = itertools.product(range(2, 23), range(1, 11), [True, False])
state_mapping = {state: i for i, state in enumerate(combinations)}

wins, draws, losses = 0, 0, 0
games = []


def get_round(
    player_sum: int,
    dealer_sum: int,
    has_ace: bool,
    action: Optional[int],
    reward: float,
    done: bool,
):
    return {
        "State": {
            "Player Sum": player_sum,
            "Dealer Sum": dealer_sum,
            "Player Has Ace": has_ace,
        },
        "Action": action,
        "Reward": reward,
        "Game Done": done,
    }


if view == random_play:
    if st.button("Run Random Gameplay!"):
        for game_number in range(number_of_games):
            observation = environment.reset()
            player_sum, dealer_sum, has_ace = observation
            done = False
            game_round = 0
            game = {
                f"Round_{game_round}": get_round(
                    player_sum, dealer_sum, has_ace, None, 0.0, done
                )
            }
            while not done:
                action = environment.action_space.sample()
                next_observation, reward, done, info = environment.step(action)
                player_sum, dealer_sum, has_ace = next_observation

                game_round += 1
                game[f"Round_{game_round}"] = get_round(
                    player_sum, dealer_sum, has_ace, action, reward, done
                )
                if done:
                    if reward == 1.0:
                        wins += 1
                    elif reward == 0.0:
                        draws += 1
                    elif reward == -1.0:
                        losses += 1
            games.append(game)
        st.write(f"{wins = }, {draws = }, {losses = }")
        win_percentage = wins / number_of_games
        st.metric("Win Percentage", win_percentage * 100)
        with st.expander("Show Replay of First 10 Games"):
            st.write(games[:10])
    else:
        st.warning("Press 'Run Random Gameplay' to continue!")
elif view == human_play:
    if st.button("Run Human Gameplay!"):
        for game_number in range(number_of_games):
            observation = environment.reset()
            player_sum, dealer_sum, has_ace = observation
            done = False
            game_round = 0
            game = {
                f"Round_{game_round}": get_round(
                    player_sum, dealer_sum, has_ace, None, 0.0, done
                )
            }
            while not done:
                if player_sum < 17:
                    action = 1
                else:
                    action = 0
                next_observation, reward, done, info = environment.step(action)
                player_sum, dealer_sum, has_ace = next_observation

                game_round += 1
                game[f"Round_{game_round}"] = get_round(
                    player_sum, dealer_sum, has_ace, action, reward, done
                )
                if done:
                    if reward == 1.0:
                        wins += 1
                    elif reward == 0.0:
                        draws += 1
                    elif reward == -1.0:
                        losses += 1
            games.append(game)
        st.write(f"{wins = }, {draws = }, {losses = }")
        win_percentage = wins / number_of_games
        st.metric("Win Percentage", win_percentage * 100)
        with st.expander("Show Replay of First 10 Games"):
            st.write(games[:10])
    else:
        st.warning("Press 'Run Human Gameplay' to continue!")
elif view == trained_play:
    q_table = np.zeros([len(state_mapping), environment.action_space.n])
    if st.button("Run Q-Learned Gameplay!"):
        with st.spinner("Training Q-Learned Agent"):
            for i in range(training_steps):
                observation = environment.reset()
                done = False
                player_sum, dealer_sum, has_ace = observation

                while not done:
                    if player_sum > 22:
                        observation = (22, dealer_sum, has_ace)
                    observation_index = state_mapping[observation]
                    if random.uniform(0, 1) < epsilon:
                        action = environment.action_space.sample()
                    else:
                        action = np.argmax(q_table[observation_index])

                    next_observation, reward, done, info = environment.step(action)
                    if (player_sum <= 11 and action == 1) or (
                        player_sum >= 17 and action == 0
                    ):
                        reward = 1
                    elif (player_sum <= 11 and action == 0) or (
                        player_sum >= 17 and action == 1
                    ):
                        reward = -1
                    player_sum, dealer_sum, has_ace = next_observation
                    if player_sum > 22:
                        next_observation = (22, dealer_sum, has_ace)
                    next_observation_index = state_mapping[next_observation]
                    previous_q_score = q_table[observation_index, action]
                    next_q_score = np.max(q_table[next_observation_index])

                    new_value = previous_q_score + alpha * (
                        reward + gamma * next_q_score - previous_q_score
                    )
                    q_table[observation_index, action] = new_value

                    observation = next_observation
        for game_number in range(number_of_games):
            observation = environment.reset()
            player_sum, dealer_sum, has_ace = observation
            if player_sum > 22:
                observation = (22, dealer_sum, has_ace)
            observation_index = state_mapping[observation]
            done = False
            game_round = 0
            game = {
                f"Round_{game_round}": get_round(
                    player_sum, dealer_sum, has_ace, None, 0.0, done
                )
            }
            while not done:
                action = np.argmax(q_table[observation_index])
                next_observation, reward, done, info = environment.step(action)
                player_sum, dealer_sum, has_ace = next_observation
                if player_sum > 22:
                    next_observation = (22, dealer_sum, has_ace)
                observation_index = state_mapping[next_observation]
                game_round += 1
                game[f"Round_{game_round}"] = get_round(
                    player_sum, dealer_sum, has_ace, action, reward, done
                )
                if done:
                    if reward == 1.0:
                        wins += 1
                    elif reward == 0.0:
                        draws += 1
                    elif reward == -1.0:
                        losses += 1
            games.append(game)
        st.write(f"{wins = }, {draws = }, {losses = }")
        win_percentage = wins / number_of_games
        st.metric("Win Percentage", win_percentage * 100)
        with st.expander("Show Replay of First 10 Games"):
            st.write(games[:10])
        with st.expander("Full Q Table"):
            stick_vs_hit = []
            for state, scores in zip(state_mapping.keys(), q_table):
                player_sum, dealer_sum, player_has_ace = state
                stick_score, hit_score = scores
                stick_vs_hit.append(
                    {
                        "Player Sum": player_sum,
                        "Dealer Sum": dealer_sum,
                        "Player Has Ace": player_has_ace,
                        "Stick": stick_score,
                        "Hit": hit_score,
                    }
                )
            st.write(pd.DataFrame(stick_vs_hit))
environment.close()

st.write(
    """## Take it further:

- Use a grid search or genetic algorithm to optimize the training Hyperparameters
- Modify the Blackjack learning environment by simplifying the state or adding more reward heuristics
- Explore different RL algorithms on Blackjack
- Explore different RL environments such as [Atari Games](https://www.gymlibrary.ml/environments/atari/) and [2D Control Tasks](https://www.gymlibrary.ml/environments/box2d/)
"""
)

if st.checkbox("Show Code (~240 lines)"):
    with open(__file__, "r") as f:
        st.code(f.read())
