import numpy as np
import pytest

from crazyflie_model.drone import Drone
from crazyflie_model.utils import euler_to_quaternion


@pytest.fixture
def drone():
    return Drone()


def make_state(pos, vel, q, omega):
    # Tworzy pełny 13-elementowy wektor stanu drona
    return np.concatenate([pos, vel, q, omega])


def test_hover(drone):
    pos = np.zeros(3)
    vel = np.zeros(3)
    omega = np.zeros(3)
    q = euler_to_quaternion(0.0, 0.0, 0.0)

    # Zarówno stan aktualny, jak i zadany są identyczne (zawis w miejscu)
    curr_state = make_state(pos, vel, q, omega)
    ref_state = make_state(pos, vel, q, omega)

    ref_thrust = drone.mass * drone.gravity
    ref_alpha = np.zeros(3)
    yaw_ref = 0.0

    thrust, torque = drone.mellinger_control(
        curr_state=curr_state,
        ref_state=ref_state,
        ref_thrust=ref_thrust,
        ref_alpha=ref_alpha,
        yaw_ref=yaw_ref,
    )

    # W idealnym zawisie ciąg musi równoważyć siłę ciężkości
    assert np.isclose(thrust, drone.mass * drone.gravity, atol=1e-3), (
        f'Ciąg w zawisie ({thrust}) powinien równać się m*g ({drone.mass * drone.gravity})'
    )
    # Momenty sił powinny być zerowe
    assert np.allclose(torque, np.zeros(3), atol=1e-6), f'Momenty powinny być 0, są: {torque}'


def test_position_error_increases_thrust(drone):
    pos_curr = np.array([0.0, 0.0, 0.0])
    pos_ref = np.array([0.0, 0.0, 1.0])  # Cel znajduje się wyżej niż dron

    vel = np.zeros(3)
    omega = np.zeros(3)
    q = euler_to_quaternion(0.0, 0.0, 0.0)

    curr_state = make_state(pos_curr, vel, q, omega)
    ref_state = make_state(pos_ref, vel, q, omega)

    ref_thrust = drone.mass * drone.gravity
    ref_alpha = np.zeros(3)
    yaw_ref = 0.0

    thrust, _ = drone.mellinger_control(
        curr_state=curr_state,
        ref_state=ref_state,
        ref_thrust=ref_thrust,
        ref_alpha=ref_alpha,
        yaw_ref=yaw_ref,
    )

    # Skoro cel jest wyżej, regulator musi wygenerować ciąg większy niż ciężar drona
    assert thrust > drone.mass * drone.gravity, (
        f'Ciąg ({thrust}) powinien być większy niż m*g ({drone.mass * drone.gravity}) przy locie w górę'
    )


def test_orientation_error_generates_torque(drone):
    pos = np.zeros(3)
    vel = np.zeros(3)
    omega = np.zeros(3)

    # Wprowadzamy błąd orientacji (aktualna rotacja roll = 0.2 rad, zadana = 0.0)
    q_curr = euler_to_quaternion(0.2, 0.0, 0.0)
    q_ref = euler_to_quaternion(0.0, 0.0, 0.0)

    curr_state = make_state(pos, vel, q_curr, omega)
    ref_state = make_state(pos, vel, q_ref, omega)

    ref_thrust = drone.mass * drone.gravity
    ref_alpha = np.zeros(3)
    yaw_ref = 0.0

    _, torque = drone.mellinger_control(
        curr_state=curr_state,
        ref_state=ref_state,
        ref_thrust=ref_thrust,
        ref_alpha=ref_alpha,
        yaw_ref=yaw_ref,
    )

    # Spodziewamy się reakcji w postaci wygenerowania momentu siły do skontrowania przechyłu
    assert not np.allclose(torque, np.zeros(3), atol=1e-6), (
        'Regulator powinien wygenerować moment obrotowy'
    )


def test_no_nan_outputs(drone):
    pos = np.zeros(3)
    vel = np.zeros(3)
    omega = np.zeros(3)
    q = euler_to_quaternion(0.0, 0.0, 0.0)

    curr_state = make_state(pos, vel, q, omega)
    ref_state = make_state(pos, vel, q, omega)

    ref_thrust = drone.mass * drone.gravity
    ref_alpha = np.zeros(3)
    yaw_ref = 0.0

    thrust, torque = drone.mellinger_control(
        curr_state=curr_state,
        ref_state=ref_state,
        ref_thrust=ref_thrust,
        ref_alpha=ref_alpha,
        yaw_ref=yaw_ref,
    )

    assert np.isfinite(thrust), 'Ciąg zawiera wartości NaN lub Inf'
    assert np.all(np.isfinite(torque)), 'Momenty obrotowe zawierają wartości NaN lub Inf'
