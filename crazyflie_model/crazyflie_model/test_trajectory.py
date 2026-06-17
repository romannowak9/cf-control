import numpy as np

from crazyflie_model.mellinger_trajectory import MellingerTrajectory

TOL = 1e-6


def create_mock_trajectory():
    p0 = [0.0, 0.0, 0.0]
    pT = [2.0, -1.0, 3.0]
    yaw0 = 0.0
    yawT = np.pi / 2
    T_flight = 5.0
    return MellingerTrajectory(p0, pT, yaw0, yawT, T_flight)


def test_start_conditions():
    traj = create_mock_trajectory()
    pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_acc = traj.evaluate(0.0)

    assert np.allclose(pos, traj.p0, atol=TOL), f'Zła pozycja startowa: {pos}'
    assert np.allclose(vel, np.zeros(3), atol=TOL), f'Prędkość na starcie nie jest zerowa: {vel}'
    assert np.allclose(acc, np.zeros(3), atol=TOL), (
        f'Przyspieszenie na starcie nie jest zerowe: {acc}'
    )
    assert np.allclose(jerk, np.zeros(3), atol=TOL), f'Jerk na starcie nie jest zerowy: {jerk}'
    assert np.all(np.isfinite(snap)), 'Snap zawiera nieliczbowe wartości'
    assert np.isclose(yaw, traj.yaw0, atol=TOL), f'Zły yaw startowy: {yaw}'
    assert np.isclose(yaw_dot, 0.0, atol=TOL), 'Prędkość yaw na starcie nie jest zerowa'
    assert np.isclose(yaw_acc, 0.0, atol=TOL), 'Przyspieszenie yaw na starcie nie jest zerowe'


def test_end_conditions():
    traj = create_mock_trajectory()
    pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_acc = traj.evaluate(traj.T_flight)

    assert np.allclose(pos, traj.pT, atol=TOL), f'Nie osiągnięto pozycji docelowej: {pos}'
    assert np.allclose(vel, np.zeros(3), atol=TOL), f'Prędkość na końcu nie wygasła: {vel}'
    assert np.allclose(acc, np.zeros(3), atol=TOL), f'Przyspieszenie na końcu nie wygasło: {acc}'
    assert np.allclose(jerk, np.zeros(3), atol=TOL), f'Jerk na końcu nie wygasł: {jerk}'
    assert np.allclose(snap, np.zeros(3), atol=TOL), f'Snap na końcu nie wygasł: {snap}'
    assert np.isclose(yaw, traj.yawT, atol=TOL), f'Nie osiągnięto zadanego yaw: {yaw}'
    assert np.isclose(yaw_dot, 0.0, atol=TOL), 'Prędkość yaw na końcu nie wygasła'
    assert np.isclose(yaw_acc, 0.0, atol=TOL), 'Przyspieszenie yaw na końcu nie wygasło'


def test_midpoint_symmetry():
    traj = create_mock_trajectory()
    t_mid = traj.T_flight / 2.0
    pos, _, _, _, _, yaw, _, _ = traj.evaluate(t_mid)

    expected_pos = traj.p0 + 0.5 * (traj.pT - traj.p0)
    expected_yaw = traj.yaw0 + 0.5 * (traj.yawT - traj.yaw0)

    assert np.allclose(pos, expected_pos, atol=TOL), (
        f'Środek pozycji niesymetryczny: {pos} vs {expected_pos}'
    )
    assert np.isclose(yaw, expected_yaw, atol=TOL), (
        f'Środek obrotu yaw niesymetryczny: {yaw} vs {expected_yaw}'
    )


def test_post_flight_hover():
    traj = create_mock_trajectory()
    t_over = traj.T_flight + 2.0
    pos, vel, acc, _, _, yaw, yaw_dot, _ = traj.evaluate(t_over)

    assert np.allclose(pos, traj.pT, atol=TOL), 'Po zakończeniu lotu pozycja dryfuje'
    assert np.allclose(vel, np.zeros(3), atol=TOL), 'Po zakończeniu lotu prędkość nie jest zerem'
    assert np.allclose(acc, np.zeros(3), atol=TOL), (
        'Po zakończeniu lotu przyspieszenie nie jest zerem'
    )
    assert np.isclose(yaw, traj.yawT, atol=TOL), 'Po zakończeniu lotu yaw dryfuje'
    assert np.isclose(yaw_dot, 0.0, atol=TOL), 'Po zakończeniu lotu yaw_dot nie jest zerem'


def test_negative_time_handling():
    traj = create_mock_trajectory()
    pos_neg, _, _, _, _, yaw_neg, _, _ = traj.evaluate(-5.0)
    pos_zero, _, _, _, _, yaw_zero, _, _ = traj.evaluate(0.0)

    assert np.allclose(pos_neg, pos_zero, atol=TOL), (
        'Ujemny czas nie pokrywa się z t=0 dla pozycji'
    )
    assert np.isclose(yaw_neg, yaw_zero, atol=TOL), 'Ujemny czas nie pokrywa się z t=0 dla yaw'


def main():
    tests = [
        ('test_start_conditions', test_start_conditions),
        ('test_end_conditions', test_end_conditions),
        ('test_midpoint_symmetry', test_midpoint_symmetry),
        ('test_post_flight_hover', test_post_flight_hover),
        ('test_negative_time_handling', test_negative_time_handling),
    ]

    print('=== URUCHAMIANIE TESTÓW GENERATORA TRAJEKTORII MELLINGERA ===\n')
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
            print(f'\n--- Szczegóły błędu w asercji {name} ---')
            print(e)
            print('-' * 50 + '\n')
            any_failed = True

    print('\n=== PODSUMOWANIE ===')
    print(f'Zaliczone testy: {passed_count}/{len(tests)}')

    if any_failed:
        print('Wynik: NIEKTÓRE TESTY TRAJEKTORII ZAKOŃCZYŁY SIĘ NIEPOWODZENIEM. 🛑')
        exit(1)
    else:
        print('Wynik: WSZYSTKIE TESTY TRAJEKTORII ZALICZONE! 🎉')
        exit(0)


if __name__ == '__main__':
    main()
