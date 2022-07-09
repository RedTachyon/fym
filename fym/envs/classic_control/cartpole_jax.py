"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
Implementation heavily based on gym
"""
from functools import partial
from typing import Optional

# import pygame
from jax.random import PRNGKey
from numpy.random import Generator
import jax
import jax.numpy as jnp

import fym
import numpy as np
# from numpy.typing import NDArray

from flax import struct
from fym.core import State, Time, Embedding, Action

VECTORIZE = True

class CartPoleEnv(fym.SARLE[jnp.ndarray, int]):
    """
    Jax version of the classic cart-pole system implemented by Rich Sutton et al.
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

    vmap = jax.vmap if VECTORIZE else lambda x: x

    @classmethod
    @partial(vmap, in_axes=(None, 0))
    @partial(jax.jit, static_argnums=(0,))
    def initial(cls, rng: Optional[jax.random.PRNGKey] = None) -> State:
        if rng is None:
            rng = jax.random.PRNGKey(np.random.default_rng().integers(0, 2 ** 32 - 1))
        return jax.random.uniform(key=rng, minval=-0.05, maxval=0.05, shape=(4,))

    @classmethod
    @partial(vmap, in_axes=(None, 0, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def transition(
            cls, time: Time, state: State, action: Action, rng: Optional[PRNGKey] = None
    ) -> State:
        x, x_dot, theta, theta_dot = state
        force = jnp.sign(action - 0.5) * cls.force_mag
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
                       force + cls.polemass_length * theta_dot ** 2 * sintheta
               ) / cls.total_mass
        thetaacc = (cls.gravity * sintheta - costheta * temp) / (
                cls.length * (4.0 / 3.0 - cls.masspole * costheta ** 2 / cls.total_mass)
        )
        xacc = temp - cls.polemass_length * thetaacc * costheta / cls.total_mass

        x = x + cls.tau * x_dot
        x_dot = x_dot + cls.tau * xacc
        theta = theta + cls.tau * theta_dot
        theta_dot = theta_dot + cls.tau * thetaacc

        return jnp.array((x, x_dot, theta, theta_dot), dtype=jnp.float32)

    @classmethod
    @partial(vmap, in_axes=(None, 0, 0, None, None))
    @partial(jax.jit, static_argnums=(0,))
    def reward(
            cls,
            time: Time,
            state: State,
            action: Optional[Action] = None,
            next_state: Optional[State] = None,
    ) -> float:
        x, x_dot, theta, theta_dot = state

        done = (x < -cls.x_threshold) \
               | (x > cls.x_threshold) \
               | (theta < -cls.theta_threshold_radians) \
               | (theta > cls.theta_threshold_radians)

        return jax.lax.cond(done, lambda: 0.0, lambda: 1.0)

    @classmethod
    @partial(vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def embedding(cls, time: Time, state: State) -> Embedding:
        return jnp.append(state, time / cls.max_time)

    @classmethod
    @partial(vmap, in_axes=(None, 0))
    @partial(jax.jit, static_argnums=(0,))
    def terminal(cls, state: State) -> bool:
        x, x_dot, theta, theta_dot = state

        done = (x < -cls.x_threshold) \
               | (x > cls.x_threshold) \
               | (theta < -cls.theta_threshold_radians) \
               | (theta > cls.theta_threshold_radians)

        return done

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

    # @classmethod
    # def render(cls, time: Time, state: State, **kwargs) -> np.ndarray:
    #     from fym.envs.classic_control import rendering
    #
    #     screen_width = 600
    #     screen_height = 400
    #
    #     world_width = cls.x_threshold * 2
    #     scale = screen_width / world_width
    #     carty = 100  # TOP OF CART
    #     polewidth = 10.0
    #     polelen = scale * (2 * cls.length)
    #     cartwidth = 50.0
    #     cartheight = 30.0
    #
    #     viewer = rendering.Viewer(screen_width, screen_height)
    #     l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
    #     axleoffset = cartheight / 4.0
    #     cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #     carttrans = rendering.Transform()
    #     cart.add_attr(carttrans)
    #     viewer.add_geom(cart)
    #     l, r, t, b = (
    #         -polewidth / 2,
    #         polewidth / 2,
    #         polelen - polewidth / 2,
    #         -polewidth / 2,
    #     )
    #     pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #     pole.set_color(0.8, 0.6, 0.4)
    #     poletrans = rendering.Transform(translation=(0, axleoffset))
    #     pole.add_attr(poletrans)
    #     pole.add_attr(carttrans)
    #     viewer.add_geom(pole)
    #     axle = rendering.make_circle(polewidth / 2)
    #     axle.add_attr(poletrans)
    #     axle.add_attr(carttrans)
    #     axle.set_color(0.5, 0.5, 0.8)
    #     viewer.add_geom(axle)
    #     track = rendering.Line((0, carty), (screen_width, carty))
    #     track.set_color(0, 0, 0)
    #     viewer.add_geom(track)
    #
    #     _pole_geom = pole
    #
    #     # Edit the pole polygon vertex
    #     pole = _pole_geom
    #     l, r, t, b = (
    #         -polewidth / 2,
    #         polewidth / 2,
    #         polelen - polewidth / 2,
    #         -polewidth / 2,
    #     )
    #     pole.v = [(l, b), (l, t), (r, t), (r, b)]
    #
    #     x = state
    #     cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
    #     carttrans.set_translation(cartx, carty)
    #     poletrans.set_rotation(-x[2])
    #
    #     img = viewer.render(return_rgb_array=True)
    #     viewer.close()
    #
    #     return img
