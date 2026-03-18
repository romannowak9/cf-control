import numpy as np
from config import *


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def euler2rotation_matrix(roll, pitch, yaw):
    """
    Konwertuje kąty Eulera (roll, pitch, yaw) na macierz rotacji 3x3.
    Kolejność rotacji: Z (yaw) -> Y (pitch) -> X (roll)

    Parametry:
        roll  - obrót wokół osi X [rad]
        pitch - obrót wokół osi Y [rad]
        yaw   - obrót wokół osi Z [rad]

    Zwraca:
        3x3 macierz rotacji R
    """
    # Rotacja wokół X (roll)
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

    # Rotacja wokół Y (pitch)
    Ry = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    )

    # Rotacja wokół Z (yaw)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    # Złożona rotacja: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


class Drone:
    def __init__(self, mass=MASS, inertia_matrix=J, init_pose=INIT_POSE, init_vel=INIT_VEL):
        """
        Params:
        - mass: float
        - inertia_matrix: np.ndarray
        """

        self.mass = mass
        self.J = inertia_matrix
        self.init_pose = init_pose
        self.init_vel = init_vel
        self.init_v = init_vel[:2]
        self.init_r = init_pose[3:]
        self.init_R = euler2rotation_matrix(*init_pose[3:])
        self.init_omega = init_vel[3:]

    def forward(self, thrust, torque):
        r_dot = v
        v_dot = G + 1 / self.mass
        q = np.quaternion(1, 0, 1, 0)


def motor_vel_to_model_input(self, motor_vel):
    pass


def main():
    print('XD')


if __name__ == '__main__':
    main()
