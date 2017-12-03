"""
Simplified policy iteration algorithm.
"""

from environment import *
import numpy as np

env = Environment()
discount_rate = 0.9
state_values = {State.HIGH: 0.0, State.LOW: 0.0}

def find_action_values(state):
    action_values = np.zeros(len(env.actions))
    for action in env.actions:
        for next_state in env.states:
            state_transition_prob = env.state_transitions[(state, action, next_state)][0]
            reward = env.state_transitions[(state, action, next_state)][1]
            action_values[action.value] += state_transition_prob * (reward + discount_rate * state_values[next_state])
    return action_values

def find_best_action_value(state):
    action_values = find_action_values(state)
    return max(action_values)

def find_best_action(state):
    action_values = find_action_values(state)
    return Action(np.argmax(action_values))

def value_iteration():
    while True:
        delta = 0
        for state in env.states:
            previous_state_value = state_values[state]
            state_values[state] = find_best_action_value(state)
            delta = max(delta, abs(previous_state_value - state_values[state]))

        if delta < 0.1: # stopping condition
            break


policy = {State.HIGH: {Action.MOVE: 0, Action.RECHARGE: 0},
           State.LOW: {Action.MOVE: 0, Action.RECHARGE: 0}}

def run_deterministic_policy():
    for state in env.states:
        best_action = find_best_action(state)
        policy[state][best_action] = 1.0

    print("Policy in state {}: {}".format(State.HIGH.name, policy[State.HIGH]))
    print("Policy in state {}: {}".format(State.LOW.name, policy[State.LOW]))

value_iteration()
run_deterministic_policy()