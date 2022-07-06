from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Generic, TypeVar, Iterable, Any

import numpy as np
from numpy.random import Generator

# State = dict[str, np.ndarray]
# Action = np.ndarray
Embedding = np.ndarray
Time = int  # Time == 0 => Infinite time horizon

State = TypeVar("State")
AgentState = TypeVar("AgentState")
AgentID = TypeVar("AgentID")
Action = TypeVar("Action")
MaybeAction = Optional[Action]
Observation = TypeVar("Observation")

FullState = tuple[State, Iterable[AgentState]]

Space = TypeVar("Space")


class BaseRLE(abc.ABC, Generic[AgentID, State, Action, Observation]):
    num_agents: AgentID | list[AgentID]
    state_space: State
    agent_state_space: AgentState
    action_spaces: dict[AgentID, Action]
    observation_spaces: dict[AgentID, Observation]
    max_time: int

    # Implement either single_observation or observation
    def single_observation(
        self, agent_id: AgentID, time: Time, state: FullState
    ) -> Observation:
        return self.observation(time, state)[agent_id]

    def observation(self, time: Time, state: FullState) -> dict[AgentID, Observation]:
        if isinstance(self.num_agents, int):
            return {
                agent_id: self.single_observation(agent_id, time, state)
                for agent_id in range(self.num_agents)
            }
        else:
            return {
                agent_id: self.single_observation(agent_id, time, state)
                for agent_id in self.num_agents
            }

    @abc.abstractmethod
    def next_agent(
        self, agent_id: AgentID, time: Time, state: FullState
    ) -> Iterable[AgentID]:
        pass

    @abc.abstractmethod
    def embedding(self, observation: Observation) -> Embedding:
        pass

    # Implement either these two methods...
    def environment(
        self,
        time: Time,
        state: FullState,
        actions: dict[AgentID, MaybeAction],
        rng: Optional[Generator],
    ) -> State:
        return self.transition(time, state, actions, rng)[0]

    def dynamics(
        self,
        time: Time,
        state: FullState,
        actions: dict[AgentID, MaybeAction],
        rng: Optional[Generator],
    ) -> Iterable[AgentState]:
        return self.transition(time, state, actions, rng)[1]

    # ... or this one.
    def transition(
        self,
        time: Time,
        state: FullState,
        actions: dict[AgentID, MaybeAction],
        rng: Optional[Generator],
    ) -> FullState:
        return (
            self.environment(time, state, actions, rng),
            self.dynamics(time, state, actions, rng),
        )

    @abc.abstractmethod
    def reward(
        self,
        time: Time,
        state: FullState,
        actions: dict[AgentID, MaybeAction],
        next_state: FullState,
    ) -> dict[AgentID, float]:
        pass

    @abc.abstractmethod
    def initial(self, rng: Optional[Generator]) -> FullState:
        pass

    @abc.abstractmethod
    def terminal(self, state: FullState) -> bool:
        pass

    # Helper functions

    def render(self, state: FullState) -> Any:
        pass

    def timeout(self, time: Time) -> bool:
        return 0 < self.max_time <= time


class SARLE(abc.ABC, Generic[State, Action]):
    state_space: State
    action_space: Action
    max_time: Time

    @classmethod
    @abc.abstractmethod
    def embedding(cls, time: Time, state: State) -> Embedding:
        pass

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
    def initial(cls, rng: Optional[Generator] = None) -> State:
        pass

    @classmethod
    @abc.abstractmethod
    def terminal(cls, state: State) -> bool:
        pass

    # Helper methods

    @classmethod
    def render(cls, time: Time, state: State):
        pass

    @classmethod
    def timeout(cls, time: Time) -> bool:
        return 0 < cls.max_time <= time


class TimeIndependentRLE(abc.ABC, Generic[State, Action]):
    max_time: Time = 0

    @classmethod
    @abc.abstractmethod
    def transition(
        cls, state: State, action: Action, rng: Optional[Generator] = None
    ) -> State:
        pass

    @classmethod
    @abc.abstractmethod
    def reward(
        cls,
        state: State,
        action: Optional[Action] = None,
        next_state: Optional[State] = None,
    ) -> float:
        pass

    @classmethod
    @abc.abstractmethod
    def representation(cls, state: State) -> Embedding:
        pass

    @classmethod
    @abc.abstractmethod
    def terminal(cls, state: State) -> bool:
        pass

    @classmethod
    @abc.abstractmethod
    def initial(cls, rng: Optional[Generator] = None) -> State:
        pass

    @classmethod
    def render(cls, state: State):
        pass


class MARLE(abc.ABC, Generic[State, Action, AgentID]):
    agents: Iterable[AgentID]
    max_time: Time

    @classmethod
    @abc.abstractmethod
    def transition(
        cls,
        time: Time,
        state: State,
        action: Iterable[Action],
        rng: Optional[Generator] = None,
    ) -> State:
        pass

    @classmethod
    @abc.abstractmethod
    def reward(
        cls,
        agent_id: int,
        time: Time,
        state: State,
        action: Iterable[Action],
        next_state: Optional[State] = None,
    ) -> float:
        pass

    @classmethod
    @abc.abstractmethod
    def representation(cls, agent_id: int, time: Time, state: State) -> Embedding:
        pass

    @classmethod
    @abc.abstractmethod
    def terminal(cls, state: State) -> bool:
        pass

    @classmethod
    @abc.abstractmethod
    def initial(cls, rng: Optional[Generator] = None) -> State:
        pass

    @classmethod
    def render(cls, time: Time, state: State):
        pass
