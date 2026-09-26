"""Shared fixtures for the correctness / contract test suite."""

from __future__ import annotations

import numpy as np
import pytest

from dbn.reinforcement.envs import CartPole, Pendulum


@pytest.fixture
def cartpole():
    return CartPole(n_envs=1, seed=0)


@pytest.fixture
def cartpole_vec():
    return CartPole(n_envs=4, seed=0)


@pytest.fixture
def pendulum():
    return Pendulum(n_envs=1, seed=0)


@pytest.fixture
def pendulum_vec():
    return Pendulum(n_envs=4, seed=0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
