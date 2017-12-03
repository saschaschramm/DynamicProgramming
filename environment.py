"""
A simple environment to study reinforcement learning problems similar to the environment defined in the book
Reinforcement Learning: An Introduction.

Definition of the environment:
- A robot has the job to clean the room.
- The robot gets a reward when it moves.
- If the energy level is high the robot can move without risk of depleting the battery. With probability 1-alpha the
energy level reduces to low.
- If the energy level is low the robot depletes the battery with probability 1-beta. In this case the robot must be
rescued which produces a negative reward.
"""

from enum import Enum
import random

class Action(Enum):
    MOVE = 0
    RECHARGE = 1

    def __str__(self):
        return self.name

class State(Enum):
    HIGH = 0
    LOW = 1

    def __str__(self):
        return self.name

class Environment(object):

    def __init__(self, alpha = 0.2, beta = 0.7, reward_move = 0.4):
        self.alpha = alpha
        self.beta = beta
        self.states = [State.HIGH, State.LOW]
        self.actions = [Action.MOVE, Action.RECHARGE]

        self.state_transitions = {
            (State.HIGH, Action.MOVE, State.HIGH): (alpha, reward_move),
            (State.HIGH, Action.MOVE, State.LOW): (1 - alpha, reward_move),
            (State.LOW, Action.MOVE, State.HIGH): (1 - beta, -3),
            (State.LOW, Action.MOVE, State.LOW): (beta, reward_move),
            (State.HIGH, Action.RECHARGE, State.HIGH): (1, 0),
            (State.HIGH, Action.RECHARGE, State.LOW): (0, 0),
            (State.LOW, Action.RECHARGE, State.HIGH): (1, 0),
            (State.LOW, Action.RECHARGE, State.LOW): (0, 0),
        }

    def _random_state(self, prob_high):
        if random.uniform(0, 1) < prob_high:
            return State.HIGH
        else:
            return State.LOW

    def step(self, last_state, action):
        state = None
        if last_state == State.HIGH and action == Action.MOVE:
            state = self._random_state(self.alpha)
        elif last_state == State.LOW and action == Action.MOVE:
            state = self._random_state(1 - self.beta)
        elif last_state == State.HIGH and Action.RECHARGE:
            state = State.HIGH
        elif last_state == State.LOW and Action.RECHARGE:
            state = State.HIGH
        else:
            NotImplementedError

        reward = self.state_transitions[(last_state, action, state)][1]
        return state, reward