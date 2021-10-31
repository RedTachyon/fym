"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
Implementation heavily based on gym
"""

from typing import Optional

# import pygame
from numpy.random import Generator

import fym
import numpy as np
from numpy.typing import NDArray

from fym.core import State, Time, Representation, Action


class CartPoleEnv(fym.BaseRLE[NDArray[np.float32], int]):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    # Constants
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5
    polemass_length = masspole + length
    force_mag = 10.0
    tau = 0.02
    kinematics_integrator = "euler"
    theta_threshold_radians = 12 * 2 * np.pi / 360
    x_threshold = 2.4

    @classmethod
    def transition(
        cls, time: Time, state: State, action: Action, rng: Optional[Generator] = None
    ) -> State:
        x, x_dot, theta, theta_dot = state
        force = np.sign(action - 0.5) * cls.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + cls.polemass_length * theta_dot ** 2 * sintheta
        ) / cls.total_mass
        thetaacc = (cls.gravity * sintheta - costheta * temp) / (
            cls.length * (4.0 / 3.0 - cls.masspole * costheta ** 2 / cls.total_mass)
        )
        xacc = temp - cls.polemass_length * thetaacc * costheta / cls.total_mass

        if cls.kinematics_integrator == "euler":
            x = x + cls.tau * x_dot
            x_dot = x_dot + cls.tau * xacc
            theta = theta + cls.tau * theta_dot
            theta_dot = theta_dot + cls.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + cls.tau * xacc
            x = x + cls.tau * x_dot
            theta_dot = theta_dot + cls.tau * thetaacc
            theta = theta + cls.tau * theta_dot

        return np.array((x, x_dot, theta, theta_dot), dtype=np.float32)

    @classmethod
    def reward(
        cls,
        time: Time,
        state: State,
        action: Optional[Action] = None,
        next_state: Optional[State] = None,
    ) -> float:
        x, x_dot, theta, theta_dot = state

        done = bool(
            x < -cls.x_threshold
            or x > cls.x_threshold
            or theta < -cls.theta_threshold_radians
            or theta > cls.theta_threshold_radians
        )

        if done:
            reward = 0.0
        else:
            reward = 1.0

        return reward

    @classmethod
    def representation(cls, time: Time, state: State) -> Representation:
        return np.append(state, time / cls.max_time)

    @classmethod
    def terminal(cls, time: Time, state: State) -> bool:
        x, x_dot, theta, theta_dot = state

        done = bool(
            x < -cls.x_threshold
            or x > cls.x_threshold
            or theta < -cls.theta_threshold_radians
            or theta > cls.theta_threshold_radians
            or (0 < cls.max_time <= time)
        )

        return done

    @classmethod
    def initial(cls, rng: Optional[Generator] = None) -> State:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(low=-0.05, high=0.05, size=(4,))

    # @classmethod # TODO: Some better rendering method
    # def render(cls, time: Time, state: State, **kwargs) -> NDArray[np.float32]:
    #     x, x_dot, theta, theta_dot = state
    #     x_screen = 600
    #     y_screen = 400
    #
    #     world_width = cls.x_threshold * 2
    #     scale = x_screen / world_width
    #
    #     carty = 100  # TOP OF CART
    #     polewidth = 10.0
    #     polelen = scale * (2 * cls.length)
    #     cartwidth = 50.0
    #     cartheight = 30.0
    #     screen = pygame.Surface((y_screen, x_screen))
    #     screen.fill((255, 255, 255))
    #
    #     cart = pygame.Rect(y_screen / 2, x_screen // 2 + x * scale, cartheight, cartwidth)
    #
    #     pygame.draw.rect(screen, (0, 0, 0), cart)
    #     observation = pygame.surfarray.pixels3d(screen)
    #
    #     return observation

    @classmethod
    def render(cls, time: Time, state: State, **kwargs) -> np.ndarray:
        from fym.envs.classic_control import rendering

        screen_width = 600
        screen_height = 400

        world_width = cls.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * cls.length)
        cartwidth = 50.0
        cartheight = 30.0

        viewer = rendering.Viewer(screen_width, screen_height)
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        carttrans = rendering.Transform()
        cart.add_attr(carttrans)
        viewer.add_geom(cart)
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        pole.set_color(0.8, 0.6, 0.4)
        poletrans = rendering.Transform(translation=(0, axleoffset))
        pole.add_attr(poletrans)
        pole.add_attr(carttrans)
        viewer.add_geom(pole)
        axle = rendering.make_circle(polewidth / 2)
        axle.add_attr(poletrans)
        axle.add_attr(carttrans)
        axle.set_color(0.5, 0.5, 0.8)
        viewer.add_geom(axle)
        track = rendering.Line((0, carty), (screen_width, carty))
        track.set_color(0, 0, 0)
        viewer.add_geom(track)

        _pole_geom = pole

        # Edit the pole polygon vertex
        pole = _pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        carttrans.set_translation(cartx, carty)
        poletrans.set_rotation(-x[2])

        img = viewer.render(return_rgb_array=True)
        viewer.close()

        return img
