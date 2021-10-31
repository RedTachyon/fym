from typing import Type, Optional

import gym

import fym


class GymEnv(gym.Env):
    """Kinda defeats the point, doesn't it?"""
    def __init__(self, rle: Type[fym.BaseRLE]):
        self.rle = rle
        self.rle.max_time = 0

        self.state = None
        self.np_random = None  # To be changed when seeding changes get merged

    def step(self, action):
        next_state = self.rle.transition(0, self.state, action, self.np_random)
        reward = self.rle.reward(0, self.state, action, next_state)
        done = self.rle.terminal(0, next_state)
        info = {}
        self.state = next_state
        return next_state, reward, done, info

    def reset(self, seed: Optional[int] = None):
        # super().__init__(seed)
        self.state = self.rle.initial(rng=self.np_random)
        return self.rle.representation(0, self.state)

    def render(self, mode="human"):
        return self.rle.render(0, self.state)

