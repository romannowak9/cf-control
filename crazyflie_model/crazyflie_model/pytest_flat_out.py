import os

import numpy as np
import pandas as pd
import pytest

from crazyflie_model.drone import Drone

TOL = 1e-6


def load_test_cases():
    # Dynamiczne ustalanie ścieżki do pliku CSV (obok pliku testowego)
    current_dir = os.path.dirname(os.path.abspath(__file__))
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


@pytest.mark.parametrize('row', load_test_cases(), ids=lambda r: r['test_name'])
def test_flat_outputs(row):
    # Inicjalizacja drona z dokładnymi parametrami z pliku CSV (w tym in_gravity!)
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

    # Wywołanie zaktualizowanej funkcji (odbieramy stan, sterowanie oraz ignorowane omega_dot)
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
