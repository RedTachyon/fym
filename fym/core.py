import abc
from typing import Optional, Generic, TypeVar

import numpy as np
from numpy.random import Generator

# State = dict[str, np.ndarray]
# Action = np.ndarray
Representation = np.ndarray
Time = int  # Time == 0 => Infinite time horizon

State = TypeVar("State")
Action = TypeVar("Action")


class BaseRLE(abc.ABC, Generic[State, Action]):
    max_time: Time

    @classmethod
    @abc.abstractmethod
    def transition(
        cls, time: Time, state: State, action: Action, rng: Optional[Generator] = None
    ) -> State:
        pass

    @classmethod
    @abc.abstractmethod
    def reward(
        cls,
        time: Time,
        state: State,
        action: Optional[Action] = None,
        next_state: Optional[State] = None,
    ) -> float:
        pass

    @classmethod
    @abc.abstractmethod
    def representation(cls, time: Time, state: State) -> Representation:
        pass

    @classmethod
    @abc.abstractmethod
    def terminal(cls, time: Time, state: State) -> bool:
        pass

    @classmethod
    @abc.abstractmethod
    def initial(cls, rng: Optional[Generator] = None) -> State:
        pass

    @classmethod
    def render(cls, time: Time, state: State):
        pass
