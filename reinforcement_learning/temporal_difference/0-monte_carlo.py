#!/usr/bin/env python3
import numpy as np

def monte_carlo_with_eligibility_traces(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, lambtha=0.9):
    """
    Monte Carlo algorithm with eligibility traces for value estimation.

    Args:
        env: The environment instance, assumed to follow OpenAI's gym interface.
        V: numpy.ndarray of shape (s,), the current estimate of the state-value function,
           where s is the number of states in the environment.
        policy: A function that takes a state as input and returns the next action to take.
        episodes: int, the total number of episodes to train over (default is 5000).
        max_steps: int, the maximum number of steps allowed per episode (default is 100).
        alpha: float, the learning rate for updating the value estimates (default is 0.1).
        gamma: float, the discount factor, which determines the importance of future rewards (default is 0.99).
        lambtha: float, the decay rate of eligibility traces (default is 0.9).

    Returns:
        numpy.ndarray: The updated state-value function V after training.
    """
    # Loop through each episode
    for ep in range(episodes):
        # Reset the environment to get the initial state
        state = env.reset()

        # Initialize eligibility traces as a list of zeros for each state
        eligibility_traces = [0 for _ in range(env.observation_space.n)]

        # Run the episode until completion or max steps are reached
        for step in range(max_steps):
            # Decay eligibility traces by lambtha * gamma
            eligibility_traces = [trace * lambtha * gamma for trace in eligibility_traces]

            # Increment eligibility trace for the current state
            eligibility_traces[state] += 1

            # Select an action using the policy
            action = policy(state)

            # Take the action and observe the next state and reward
            next_state, reward, done, _ = env.step(action)

            # Update the reward based on terminal conditions
            if env.desc.reshape(env.observation_space.n)[next_state] == b'H':
                reward = -1  # penalty for falling into a hole
            elif env.desc.reshape(env.observation_space.n)[next_state] == b'G':
                reward = 1  # reward for reaching the goal

            # Calculate the temporal-difference error
            delta_t = reward + gamma * V[next_state] - V[state]

            # Update the value function for the current state
            V[state] += alpha * delta_t * eligibility_traces[state]

            # If the episode ends, break the loop
            if done:
                break

            # Move to the next state
            state = next_state

    # Return the updated state-value function as a numpy array
    return np.array(V)
