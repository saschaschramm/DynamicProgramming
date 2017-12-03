"""
We determine the best policy by trying all possible (deterministic) policies.
"""

from environment import *
env = Environment()
discount_rate = 0.9

T = 2

policy_0 = {State.HIGH: {Action.MOVE: 1, Action.RECHARGE: 0},
             State.LOW: {Action.MOVE: 1, Action.RECHARGE: 0}}

policy_1 = {State.HIGH: {Action.MOVE: 0, Action.RECHARGE: 1},
             State.LOW: {Action.MOVE: 1, Action.RECHARGE: 0}}

policy_2 = {State.HIGH: {Action.MOVE: 0, Action.RECHARGE: 1},
            State.LOW:  {Action.MOVE: 0, Action.RECHARGE: 1}}

policy_3 = {State.HIGH: {Action.MOVE: 1, Action.RECHARGE: 0},
             State.LOW: {Action.MOVE: 0, Action.RECHARGE: 1}}

policies = [policy_0, policy_1, policy_2, policy_3]

def state_value_function(state, t, policy):
    state_value = 0
    for action, action_prob in policy[state].items():
        for next_state in env.states:
            state_transition_prob = env.state_transitions[(state, action, next_state)][0]
            reward = env.state_transitions[(state, action, next_state)][1]
            if t < T-1:
                state_value += action_prob * state_transition_prob * (reward + discount_rate * state_value_function(next_state, t+1, policy))
            else:
                state_value += action_prob * state_transition_prob * reward
    return state_value

print("Expected values with start state LOW:")
for index, policy in enumerate(policies):
    print("Policy {}: {}".format(index, state_value_function(State.HIGH, 0, policy)))

print("Expected values with start state HIGH:")
for index, policy in enumerate(policies):
    print("Policy {}: {}".format(index, state_value_function(State.LOW, 0, policy)))
