from typing import Optional

from numpy.random import Generator

import fym
import numpy as np
from os import path

from fym.core import State, Action, Time, Embedding


class PendulumEnv(fym.SARLE[np.ndarray, np.ndarray]):
    max_time = 200
    max_speed = 8
    max_torque = 2.0
    dt = 0.05
    gravity = 9.81
    mass = 1.0
    length = 1.0
    viewer = None

    @classmethod
    def transition(
        cls, time: Time, state: State, action: Action, rng: Optional[Generator] = None
    ) -> State:
        th, thdot = state

        u = np.clip(action, -cls.max_torque, cls.max_torque)

        newthdot = (
            thdot
            + (
                3 * cls.gravity / (2 * cls.length) * np.sin(th)
                + 3.0 / (cls.mass * cls.length ** 2) * u
            )
            * cls.dt
        )
        newthdot = np.clip(newthdot, -cls.max_speed, cls.max_speed)
        newth = th + newthdot * cls.dt

        state = np.array((newth, newthdot))

        return state

    @classmethod
    def reward(
        cls,
        time: Time,
        state: State,
        action: Optional[Action] = None,
        next_state: Optional[State] = None,
    ) -> float:
        th, thdot = state
        u = np.clip(action, -cls.max_torque, cls.max_torque)
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        return -costs

    @classmethod
    def initial(cls, rng: Optional[Generator] = None) -> State:
        if rng is None:
            rng = np.random.default_rng()
        high = np.array([np.pi, 1])

        state = rng.uniform(low=-high, high=high)
        return state

    @classmethod
    def embedding(cls, time: Time, state: State) -> Embedding:
        theta, thetadot = state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    @classmethod
    def terminal(cls, state: State) -> bool:
        return False

    @classmethod
    def render(cls, time: Time, state: State, last_u: Optional[Action] = None):
        from fym.envs.classic_control import rendering

        viewer = rendering.Viewer(500, 500)
        viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
        rod = rendering.make_capsule(1, 0.2)
        rod.set_color(0.8, 0.3, 0.3)
        pole_transform = rendering.Transform()
        rod.add_attr(pole_transform)
        viewer.add_geom(rod)
        axle = rendering.make_circle(0.05)
        axle.set_color(0, 0, 0)
        viewer.add_geom(axle)
        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = rendering.Image(fname, 1.0, 1.0)
        imgtrans = rendering.Transform()
        img.add_attr(imgtrans)

        viewer.add_onetime(img)
        pole_transform.set_rotation(state[0] + np.pi / 2)
        if last_u is not None:
            imgtrans.scale = (-last_u / 2, np.abs(last_u) / 2)

        img = viewer.render(return_rgb_array=True)
        viewer.close()
        return img


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
