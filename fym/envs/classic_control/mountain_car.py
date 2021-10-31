"""
http://incompleteideas.net/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import math
from typing import Optional, Union

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

import fym
from fym.core import State, Time, Representation, Action


class MountainCarEnv(fym.BaseRLE[NDArray[np.float32], int]):
    """
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.

    Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).

    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07

    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right

        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.

    Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.

    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.

    Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200
    """

    max_time = 200
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.5
    goal_velocity = 0.0

    force = 0.001
    gravity = 0.0025

    @classmethod
    def transition(
        cls, time: Time, state: State, action: Action, rng: Optional[Generator] = None
    ) -> State:
        position, velocity = state

        velocity += (action - 1) * cls.force + math.cos(3 * position) * (-cls.gravity)
        velocity = np.clip(velocity, -cls.max_speed, cls.max_speed)
        position += velocity
        position = np.clip(position, cls.min_position, cls.max_position)

        if position == cls.min_position and velocity < 0:
            velocity = 0

        return np.array((position, velocity), dtype=np.float32)

    @classmethod
    def reward(
        cls,
        time: Time,
        state: State,
        action: Optional[Action] = None,
        next_state: Optional[State] = None,
    ) -> float:
        return -1.0

    @classmethod
    def terminal(cls, time: Time, state: State) -> bool:
        position, velocity = state

        done = (position >= cls.goal_position and
                velocity >= cls.goal_velocity) or 0 < time <= cls.max_time

        return done

    @classmethod
    def initial(cls, rng: Optional[Generator] = None) -> State:
        if rng is None:
            rng = np.random.default_rng()

        return np.array([rng.uniform(-0.6, 0.4), 0], dtype=np.float32)

    @classmethod
    def representation(cls, time: Time, state: State) -> Representation:
        return state

    @staticmethod
    def _height(xs: Union[float, np.ndarray]):
        return np.sin(3 * xs) * 0.45 + 0.55

    @classmethod
    def render(cls, time: Time, state: State):
        from gym.envs.classic_control import rendering

        screen_width = 600
        screen_height = 400

        world_width = cls.max_position - cls.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20


        viewer = rendering.Viewer(screen_width, screen_height)
        xs = np.linspace(cls.min_position, cls.max_position, 100)
        ys = cls._height(xs)
        xys = list(zip((xs - cls.min_position) * scale, ys * scale))

        track = rendering.make_polyline(xys)
        track.set_linewidth(4)
        viewer.add_geom(track)

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        car.add_attr(rendering.Transform(translation=(0, clearance)))
        cartrans = rendering.Transform()
        car.add_attr(cartrans)
        viewer.add_geom(car)
        frontwheel = rendering.make_circle(carheight / 2.5)
        frontwheel.set_color(0.5, 0.5, 0.5)
        frontwheel.add_attr(
            rendering.Transform(translation=(carwidth / 4, clearance))
        )
        frontwheel.add_attr(cartrans)
        viewer.add_geom(frontwheel)
        backwheel = rendering.make_circle(carheight / 2.5)
        backwheel.add_attr(
            rendering.Transform(translation=(-carwidth / 4, clearance))
        )
        backwheel.add_attr(cartrans)
        backwheel.set_color(0.5, 0.5, 0.5)
        viewer.add_geom(backwheel)
        flagx = (cls.goal_position - cls.min_position) * scale
        flagy1 = cls._height(cls.goal_position) * scale
        flagy2 = flagy1 + 50
        flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
        viewer.add_geom(flagpole)
        flag = rendering.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )
        flag.set_color(0.8, 0.8, 0)
        viewer.add_geom(flag)

        pos = state[0]
        cartrans.set_translation(
            (pos - cls.min_position) * scale, cls._height(pos) * scale
        )
        cartrans.set_rotation(math.cos(3 * pos))

        viewer.close()

        return viewer.render(return_rgb_array=True)
