from typing import Type, Optional

import gym

import fym


class GymEnv(gym.Env):
    """Kinda defeats the point, doesn't it?"""

    def __init__(self, rle: Type[fym.SARLE]):
        self.rle = rle
        self.rle.max_time = 0

        self.state = None
        self.np_random = None  # To be changed when seeding changes get merged

    def step(self, action):
        next_state = self.rle.transition(0, self.state, action, self.np_random)
        reward = self.rle.reward(0, self.state, action, next_state)
        done = self.rle.terminal(next_state)
        info = {}
        self.state = next_state
        return next_state, reward, done, info

    def reset(self, seed: Optional[int] = None):
        # super().__init__(seed)
        self.state = self.rle.initial(rng=self.np_random)
        return self.rle.embedding(0, self.state)

    def render(self, mode="human"):
        return self.rle.render(0, self.state)


class TimeWrapperGymEnv(gym.Env):
    def __init__(self, rle: Type[fym.SARLE]):
        self.rle = rle
        self.time = 0

        self.state = None
        self.np_random = None  # To be changed when seeding changes get merged

    def step(self, action):
        next_state = self.rle.transition(self.time, self.state, action, self.np_random)
        reward = self.rle.reward(self.time, self.state, action, next_state)
        self.time += 1
        done = self.rle.terminal(next_state) or self.rle.timeout(self.time)
        info = {}
        self.state = next_state
        return next_state, reward, done, info

    def reset(self, seed: Optional[int] = None):
        # super().__init__(seed)
        self.state = self.rle.initial(rng=self.np_random)
        self.time = 0
        return self.rle.embedding(self.time, self.state)

    def render(self, mode="human"):
        return self.rle.render(self.time, self.state)
