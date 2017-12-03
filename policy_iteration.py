"""
The value function helps us to find a better policy. The way of finding an optimal policy is called policy iteration.
"""

from environment import *
import numpy as np

env = Environment()
discount_rate = 0.9
state_values = {State.HIGH: 0.0, State.LOW: 0.0}

# all actions are equally likely
policy = {State.HIGH: {Action.MOVE: 0.5, Action.RECHARGE: 0.5},
           State.LOW: {Action.MOVE: 0.5, Action.RECHARGE: 0.5}}

def policy_evaluation():
    def find_state_value(state):
        state_value = 0
        for action in env.actions:
            for next_state in env.states:
                state_transition_prob = env.state_transitions[(state, action, next_state)][0]
                reward = env.state_transitions[(state, action, next_state)][1]
                action_prob = policy[state][action]
                state_value += action_prob * state_transition_prob * (reward + discount_rate * state_values[next_state])
        return state_value

    while True:
        delta = 0
        for state in env.states:
            previous_state_value = state_values[state]
            state_values[state] = find_state_value(state)
            delta = max(delta, abs(previous_state_value - state_values[state]))
            print(state_values)

        if delta < 0.1:  # stopping condition
            break

    policy_improvement()

def policy_improvement():
    def find_best_action(state):
        action_values = np.zeros(len(env.actions))
        for action in env.actions:
            for next_state in env.states:
                state_transition_prob = env.state_transitions[(state, action, next_state)][0]
                reward = env.state_transitions[(state, action, next_state)][1]
                action_values[action.value] += state_transition_prob * (reward + discount_rate * state_values[next_state])
        return Action(np.argmax(action_values))

    policy_stable = True
    for state in env.states:
        action = policy[state]
        best_action = find_best_action(state)

        policy[state] = {Action.MOVE: 0, Action.RECHARGE: 0}
        policy[state][best_action] = 1

        if action != policy[state]:
            policy_stable = False

    if policy_stable:
        print("Policy in state {}: {}".format(State.HIGH.name, policy[State.HIGH]))
        print("Policy in state {}: {}".format(State.LOW.name, policy[State.LOW]))
    else:
        policy_evaluation()

policy_evaluation()

