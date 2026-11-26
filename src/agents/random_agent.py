
import random


class RandomAgent:

    def __init__(self, action_size):
        self.action_size = action_size

    def select_action(self, observation):
        action = random.randrange(self.action_size)

        return action
