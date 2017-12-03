"""
The computation of the state-value function is called policy evaluation.
"""
from environment import *


def policy_evaluation():
    env = Environment()
    discount_rate = 0.9
    state_values = {State.HIGH: 0.0, State.LOW: 0.0}

    # all actions are equally likely
    policy = {State.HIGH: {Action.MOVE: 0.5, Action.RECHARGE: 0.5},
              State.LOW: {Action.MOVE: 0.5, Action.RECHARGE: 0.5}}

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

        if delta < 0.0001:  # stopping condition
            break

policy_evaluation()


"""
Result:
- Value of state high is positiv
- Value of state low is negative
That means being in state high is better than being in state low.
"""
