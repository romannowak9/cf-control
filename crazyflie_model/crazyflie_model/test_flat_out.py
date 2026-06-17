import os

import numpy as np
import pandas as pd

from crazyflie_model.drone import Drone

TOL = 1e-6


def load_test_cases():
    # Dynamiczne ustalanie ścieżki do pliku CSV (obok pliku testowego)
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()  # Fallback na wypadek uruchamiania np. w Jupyterze

    csv_path = os.path.join(current_dir, 'trajectory_from_flat_output_test_data.csv')

    if not os.path.exists(csv_path):
        csv_path = 'trajectory_from_flat_output_test_data.csv'

    df = pd.read_csv(csv_path)
    return df.to_dict(orient='records')


def assert_close(name, computed, expected):
    computed = np.array(computed)
    expected = np.array(expected)

    assert np.allclose(computed, expected, atol=TOL), (
        f'{name} mismatch\n'
        f'computed: {computed}\n'
        f'expected: {expected}\n'
        f'diff    : {computed - expected}'
    )


def execute_single_test(row):
    # Inicjalizacja drona z dokładnymi parametrami z pliku CSV
    drone = Drone(
        mass=row['in_mass'],
        inertia_matrix=np.diag([row['in_I_xx'], row['in_I_yy'], row['in_I_zz']]),
        gravity=row['in_gravity'],
    )

    # Wejścia (Inputs) z formatu CSV
    pos = np.array([row['in_pos_x'], row['in_pos_y'], row['in_pos_z']])
    vel = np.array([row['in_vel_x'], row['in_vel_y'], row['in_vel_z']])
    acc = np.array([row['in_acc_x'], row['in_acc_y'], row['in_acc_z']])
    jerk = np.array([row['in_jerk_x'], row['in_jerk_y'], row['in_jerk_z']])
    snap = np.array([row['in_snap_x'], row['in_snap_y'], row['in_snap_z']])

    yaw = row['in_yaw']
    yaw_dot = row['in_yaw_rate']
    yaw_acc = row['in_yaw_acceleration']

    # Oczekiwane wyjścia (Expected Outputs) z formatu CSV
    expected_pos = [row['out_pos_x'], row['out_pos_y'], row['out_pos_z']]
    expected_vel = [row['out_vel_x'], row['out_vel_y'], row['out_vel_z']]
    expected_quat = [
        row['out_quat_w'],
        row['out_quat_x'],
        row['out_quat_y'],
        row['out_quat_z'],
    ]
    expected_omega = [
        row['out_omega_x'],
        row['out_omega_y'],
        row['out_omega_z'],
    ]
    expected_control = [
        row['out_thrust'],
        row['out_torque_x'],
        row['out_torque_y'],
        row['out_torque_z'],
    ]

    # Wywołanie funkcji modelu drona
    state, control, _ = drone.flat_out_state_and_control(
        pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_acc
    )

    state = np.array(state)

    # Rozbicie wyliczonego stanu na składowe
    computed_pos = state[0:3]
    computed_vel = state[3:6]
    computed_quat = state[6:10]
    computed_omega = state[10:13]

    # Asercje porównujące wyniki z bazą CSV
    assert_close('pos', computed_pos, expected_pos)
    assert_close('vel', computed_vel, expected_vel)
    assert_close('quat', computed_quat, expected_quat)
    assert_close('omega', computed_omega, expected_omega)
    assert_close('control', control, expected_control)


def main():
    test_cases = load_test_cases()
    print(f'Załadowano {len(test_cases)} przypadków testowych.\n')

    passed_count = 0
    failed = False

    for row in test_cases:
        test_name = row['test_name']
        print(f'Uruchamianie testu: {test_name:.<40}', end='')

        try:
            execute_single_test(row)
            print('SUKCES ✅')
            passed_count += 1
        except AssertionError as e:
            print('BŁĄD ❌')
            print(f"\n--- Szczegóły błędu dla testu '{test_name}' ---")
            print(e)
            print('-' * 40 + '\n')
            failed = True
            # Jeśli wolisz kontynuować testy mimo błędu, usuń poniższe 'break'
            break

    if failed:
        print(f'Testy zakończone NIEPOWODZENIEM (zaliczone: {passed_count}/{len(test_cases)})')
        exit(1)
    else:
        print(
            f'\nGratulacje! Wszystkie testy ({passed_count}/{len(test_cases)}) przeszły pomyślnie. 🎉'
        )


if __name__ == '__main__':
    main()
