import numpy as np

from crazyflie_model.drone import Drone
from crazyflie_model.utils import euler_to_quaternion


def make_state(pos, vel, q, omega):
    return np.concatenate([pos, vel, q, omega])


def test_hover():
    drone = Drone()
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

    assert np.isclose(thrust, drone.mass * drone.gravity, atol=1e-3), (
        f'Ciąg w zawisie ({thrust}) powinien równać się m*g ({drone.mass * drone.gravity})'
    )
    assert np.allclose(torque, np.zeros(3), atol=1e-6), f'Momenty powinny być 0, są: {torque}'


def test_position_error_increases_thrust():
    drone = Drone()
    pos_curr = np.array([0.0, 0.0, 0.0])
    pos_ref = np.array([0.0, 0.0, 1.0])

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

    assert thrust > drone.mass * drone.gravity, (
        f'Ciąg ({thrust}) powinien być większy niż m*g ({drone.mass * drone.gravity}) przy błędzie wysokości'
    )


def test_orientation_error_generates_torque():
    drone = Drone()
    pos = np.zeros(3)
    vel = np.zeros(3)
    omega = np.zeros(3)

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

    assert not np.allclose(torque, np.zeros(3), atol=1e-6), (
        'Regulator powinien wygenerować moment obrotowy'
    )


def test_no_nan_outputs():
    drone = Drone()
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


def main():
    tests = [
        ('test_hover', test_hover),
        ('test_position_error_increases_thrust', test_position_error_increases_thrust),
        ('test_orientation_error_generates_torque', test_orientation_error_generates_torque),
        ('test_no_nan_outputs', test_no_nan_outputs),
    ]

    print('=== URUCHAMIANIE TESTÓW REGULATORA MELLINGERA (BEZ PYTEST) ===\n')
    passed_count = 0
    any_failed = False

    for name, test_func in tests:
        print(f'Uruchamianie: {name:.<50}', end='')
        try:
            test_func()
            print('SUKCES ✅')
            passed_count += 1
        except AssertionError as e:
            print('BŁĄD ❌')
            print(f'\n--- Szczegóły błędu w {name} ---')
            print(e)
            print('-' * 50 + '\n')
            any_failed = True

    print('\n=== PODSUMOWANIE ===')
    print(f'Zaliczone testy: {passed_count}/{len(tests)}')

    if any_failed:
        print('Wynik: TESTY NIE PRZESZŁY! 🛑')
        exit(1)
    else:
        print('Wynik: WSZYSTKIE TESTY ZALICZONE! 🎉')
        exit(0)


if __name__ == '__main__':
    main()
