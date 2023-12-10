import numpy as np
import random

# Initialize parameters
states = ['A', 'B']
actions = ['move', 'stay']
gamma = 0.8  # discount factor
alpha = 0.5  # learning rate
epsilon = 0.5  # probability of choosing a random action
num_steps = 200

# Initialize Q-table: rows for states and columns for actions
Q = np.zeros((len(states), len(actions)))

# Reward function: 0 for move, 1 for stay
def reward(state, action):
    return 1 if action == 'stay' else 0

# Deterministic state transition
def next_state(current_state, action):
    if action == 'move':
        return 'B' if current_state == 'A' else 'A'
    return current_state

# Epsilon-greedy policy
def epsilon_greedy_policy(state, Q, epsilon):
    if random.random() < epsilon:
        # Random action
        return random.choice(actions)
    else:
        # Greedy action
        state_idx = states.index(state)
        return actions[np.argmax(Q[state_idx, :])]

# Q-learning algorithm
current_state = 'A'
for step in range(num_steps):
    action = epsilon_greedy_policy(current_state, Q, epsilon)
    next_state_val = next_state(current_state, action)
    reward_val = reward(current_state, action)

    # Update Q-table
    current_state_idx = states.index(current_state)
    next_state_idx = states.index(next_state_val)
    best_future_q = np.max(Q[next_state_idx, :])
    Q[current_state_idx, actions.index(action)] = (1 - alpha) * Q[current_state_idx, actions.index(action)] + \
                                                  alpha * (reward_val + gamma * best_future_q)

    # Move to next state
    current_state = next_state_val

# Display the final Q-table after 200 steps
print(Q)
