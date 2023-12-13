import numpy as np
import random

unique_states = ['A', 'B']
unique_actions = ['move', 'stay']
discount_factor = 0.8
learning_rate = 0.5
exploration_prob = 0.5
iteration_count = 200

Q_table = np.zeros((len(unique_states), len(unique_actions)))

def calculate_reward(state, action):
    return 1 if action == 'stay' else 0

def determine_next_state(current_state, action):
    if action == 'move':
        return 'B' if current_state == 'A' else 'A'
    return current_state

def choose_epsilon_greedy_action(state, Q, epsilon):
    if random.random() < epsilon:
        return random.choice(unique_actions)
    else:
        state_index = unique_states.index(state)
        return unique_actions[np.argmax(Q[state_index, :])]

active_state = 'A'
for step in range(iteration_count):
    chosen_action = choose_epsilon_greedy_action(active_state, Q_table, exploration_prob)
    next_state_value = determine_next_state(active_state, chosen_action)
    gained_reward = calculate_reward(active_state, chosen_action)

    current_state_index = unique_states.index(active_state)
    next_state_index = unique_states.index(next_state_value)
    optimal_future_q = np.max(Q_table[next_state_index, :])
    Q_table[current_state_index, unique_actions.index(chosen_action)] = \
        (1 - learning_rate) * Q_table[current_state_index, unique_actions.index(chosen_action)] + \
        learning_rate * (gained_reward + discount_factor * optimal_future_q)

    active_state = next_state_value

print(Q_table)
