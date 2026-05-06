import numpy as np
import pytest

from crazyflie_model.drone import Drone
from crazyflie_model.utils import euler_to_quaternion


@pytest.fixture
def drone():
    return Drone()


def make_state(pos, vel, q, omega):
    return np.concatenate([pos, vel, q, omega])


def test_hover(drone):
    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)
    omega = np.zeros(3)
    yaw = 0.0

    q = euler_to_quaternion(0.0, 0.0, 0.0)
    state = make_state(pos, vel, q, omega)

    thrust, torque = drone.mellinger_control(
        curr_state=state,
        pos_target=pos,
        vel_target=vel,
        yaw_target=yaw,
        acc_target=acc,
        omega_target=omega,
    )

    assert np.isclose(thrust, drone.mass * drone.gravity, atol=1e-3)
    assert np.allclose(torque, np.zeros(3), atol=1e-6)


def test_position_error_increases_thrust(drone):
    pos_curr = np.array([0.0, 0.0, 0.0])
    pos_target = np.array([0.0, 0.0, 1.0])  # chcemy wyżej

    vel = np.zeros(3)
    acc = np.zeros(3)
    omega = np.zeros(3)
    yaw = 0.0

    q = euler_to_quaternion(0.0, 0.0, 0.0)
    state = make_state(pos_curr, vel, q, omega)

    thrust, _ = drone.mellinger_control(
        curr_state=state,
        pos_target=pos_target,
        vel_target=vel,
        yaw_target=yaw,
        acc_target=acc,
        omega_target=omega,
    )

    assert thrust > drone.mass * drone.gravity


def test_orientation_error_generates_torque(drone):
    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)
    omega = np.zeros(3)
    yaw = 0.0

    # wprowadzamy błąd orientacji (roll)
    q = euler_to_quaternion(0.2, 0.0, 0.0)
    state = make_state(pos, vel, q, omega)

    _, torque = drone.mellinger_control(
        curr_state=state,
        pos_target=pos,
        vel_target=vel,
        yaw_target=yaw,
        acc_target=acc,
        omega_target=omega,
    )

    assert not np.allclose(torque, np.zeros(3))


def test_no_nan_outputs(drone):
    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)
    omega = np.zeros(3)
    yaw = 0.0

    q = euler_to_quaternion(0.0, 0.0, 0.0)
    state = make_state(pos, vel, q, omega)

    thrust, torque = drone.mellinger_control(
        curr_state=state,
        pos_target=pos,
        vel_target=vel,
        yaw_target=yaw,
        acc_target=acc,
        omega_target=omega,
    )

    assert np.isfinite(thrust)
    assert np.all(np.isfinite(torque))
