import numpy as np
import pytest

from crazyflie_model.trajectory import MellingerTrajectory

TOL = 1e-6


@pytest.fixture
def trajectory():
    # Definiujemy trajektorię testową:
    # Lot z [0, 0, 0] do [2, -1, 3] w czasie 5 sekund, obrót yaw z 0 do 90 stopni (pi/2)
    p0 = [0.0, 0.0, 0.0]
    pT = [2.0, -1.0, 3.0]
    yaw0 = 0.0
    yawT = np.pi / 2
    T_flight = 5.0
    return MellingerTrajectory(p0, pT, yaw0, yawT, T_flight)


def test_start_conditions(trajectory):
    """Testuje czy w t = 0 dron jest w p0 i ma zerowe pochodne (gładki start)."""
    pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_acc = trajectory.evaluate(0.0)

    assert np.allclose(pos, trajectory.p0, atol=TOL)
    assert np.allclose(vel, np.zeros(3), atol=TOL)
    assert np.allclose(acc, np.zeros(3), atol=TOL)
    assert np.allclose(jerk, np.zeros(3), atol=TOL)
    # Snap na starcie wielomianu 7. stopnia nie musi być zero, sprawdzamy stabilność numeryczną
    assert np.all(np.isfinite(snap))

    assert np.isclose(yaw, trajectory.yaw0, atol=TOL)
    assert np.isclose(yaw_dot, 0.0, atol=TOL)
    assert np.isclose(yaw_acc, 0.0, atol=TOL)


def test_end_conditions(trajectory):
    """Testuje czy w t = T_flight dron osiąga pT i zatrzymuje się (zawis)."""
    pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_acc = trajectory.evaluate(trajectory.T_flight)

    assert np.allclose(pos, trajectory.pT, atol=TOL)
    assert np.allclose(vel, np.zeros(3), atol=TOL)
    assert np.allclose(acc, np.zeros(3), atol=TOL)
    assert np.allclose(jerk, np.zeros(3), atol=TOL)
    assert np.allclose(snap, np.zeros(3), atol=TOL)

    assert np.isclose(yaw, trajectory.yawT, atol=TOL)
    assert np.isclose(yaw_dot, 0.0, atol=TOL)
    assert np.isclose(yaw_acc, 0.0, atol=TOL)


def test_midpoint_symmetry(trajectory):
    """Dla wielomianów Mellingera w połowie czasu lotu stan musi być idealnie w połowie drogi."""
    t_mid = trajectory.T_flight / 2.0
    pos, _, _, _, _, yaw, _, _ = trajectory.evaluate(t_mid)

    expected_pos = trajectory.p0 + 0.5 * (trajectory.pT - trajectory.p0)
    expected_yaw = trajectory.yaw0 + 0.5 * (trajectory.yawT - trajectory.yaw0)

    assert np.allclose(pos, expected_pos, atol=TOL)
    assert np.isclose(yaw, expected_yaw, atol=TOL)


def test_post_flight_hover(trajectory):
    """Testuje czy po upływie czasu lotu (t > T_flight) wartości pozostają zablokowane na celu."""
    t_over = trajectory.T_flight + 2.0
    pos, vel, acc, _, _, yaw, yaw_dot, _ = trajectory.evaluate(t_over)

    assert np.allclose(pos, trajectory.pT, atol=TOL)
    assert np.allclose(vel, np.zeros(3), atol=TOL)
    assert np.allclose(acc, np.zeros(3), atol=TOL)
    assert np.isclose(yaw, trajectory.yawT, atol=TOL)
    assert np.isclose(yaw_dot, 0.0, atol=TOL)


def test_negative_time_handling(trajectory):
    """Testuje czy podanie ujemnego czasu traktowane jest bezpiecznie jako t = 0."""
    pos_neg, _, _, _, _, yaw_neg, _, _ = trajectory.evaluate(-1.5)
    pos_zero, _, _, _, _, yaw_zero, _, _ = trajectory.evaluate(0.0)

    assert np.allclose(pos_neg, pos_zero, atol=TOL)
    assert np.isclose(yaw_neg, yaw_zero, atol=TOL)
