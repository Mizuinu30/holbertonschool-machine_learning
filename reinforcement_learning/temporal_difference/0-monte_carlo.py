#!/usr/bin/env python3
import numpy as np

def generate_episode(env, policy, max_steps):
    """
    Generates an episode by following the given policy in the environment.

    Parameters:
        env: The OpenAI environment instance.
        policy: A function that takes in a state and returns the next action to take.
        max_steps: The maximum number of steps per episode.

    Returns:
        episode: A list containing two lists: one for states and one for rewards.
    """
    # Initialize episode structure to store states and rewards
    episode = [[], []]

    # Start by resetting the environment to get the initial state
    state = env.reset()

    # Generate the episode by following the policy
    for step in range(max_steps):
        # Select an action based on the current policy
        action = policy(state)

        # Execute the action in the environment
        next_state, reward, done, _ = env.step(action)

        # Append the current state to the list of states in the episode
        episode[0].append(state)

        # Adjust rewards for specific conditions (holes and goals)
        if env.desc.reshape(env.observation_space.n)[next_state] == b'H':
            episode[1].append(-1)  # Penalty for falling into a hole
            return episode
        elif env.desc.reshape(env.observation_space.n)[next_state] == b'G':
            episode[1].append(1)  # Reward for reaching the goal
            return episode
        else:
            # Append a reward of 0 for non-terminal states
            episode[1].append(0)

        # Transition to the next state
        state = next_state

    # Return the episode if the maximum steps are reached
    return episode


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate the value function.

    Parameters:
        env: The OpenAI environment instance.
        V: numpy.ndarray of shape (s,), containing the current value estimates, where s is the number of states.
        policy: A function that takes in a state and returns the next action to take.
        episodes: int, the total number of episodes to train over (default is 5000).
        max_steps: int, the maximum number of steps per episode (default is 100).
        alpha: float, the learning rate for updating the value estimates (default is 0.1).
        gamma: float, the discount factor (default is 0.99).

    Returns:
        numpy.ndarray: The updated value estimate V after training.
    """
    # Pre-compute discount factors for the maximum episode length
    discounts = np.array([gamma ** i for i in range(max_steps)])

    # Loop through each episode
    for ep in range(episodes):
        # Generate an episode by following the policy
        episode = generate_episode(env, policy, max_steps)

        # Update the value function based on episode returns
        for i, state in enumerate(episode[0]):
            # Compute the discounted return Gt from the ith state onward
            Gt = np.sum(np.array(episode[1][i:]) * discounts[:len(episode[1][i:])])

            # Update the value function for the current state using Gt
            V[state] += alpha * (Gt - V[state])

    return V
