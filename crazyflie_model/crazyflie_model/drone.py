import numpy as np

from .config import *


def quat_multiply(q, p):
    """Mnożenie kwaternionów."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_rotate(q, v):
    """
    rotacja wektora v przez kwaternion q
    """
    v_quat = np.array([0, *v])

    return quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))[1:]


def euler_to_quaternion(roll, pitch, yaw):
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)

    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)

    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


class Drone:
    def __init__(self, mass=MASS, inertia_matrix=J, init_pose=INIT_POSE, init_vel=INIT_VEL):

        self.mass = mass
        self.J = inertia_matrix
        self.J_inv = np.linalg.inv(self.J)

        # Initial state
        self.r = np.array(init_pose[:3], dtype=float)
        self.v = np.array(init_vel[:3], dtype=float)
        self.q = euler_to_quaternion(*init_pose[3:])
        self.omega = np.array(init_vel[3:], dtype=float)

    def __state_derivative(self, state, thrust, torque):
        v = state[3:6]
        q = state[6:10]
        omega = state[10:13]

        r_dot = v

        thrust_body = np.array([0.0, 0.0, thrust])
        thrust_world = quat_rotate(q, thrust_body)

        v_dot = np.array([0.0, 0.0, -G]) + (1 / self.mass) * thrust_world

        omega_quat = np.array([0.0, omega[0] / 2.0, omega[1] / 2.0, omega[2] / 2.0])

        q_dot = quat_multiply(q, omega_quat)

        omega_dot = self.J_inv @ (torque - np.cross(omega, self.J @ omega))

        return np.concatenate([r_dot, v_dot, q_dot, omega_dot])

    def state_model(self, thrust, torque, dt):
        """
        Compute state from input values

        Returns:
        - state: np.array - (r1,r2,r3,v1,v2,v3,q1,q2,q3,q4,omega1,omega2,omega3)
        """

        state = np.concatenate([self.r, self.v, self.q, self.omega])

        k1 = self.__state_derivative(state, thrust, torque)

        k2 = self.__state_derivative(
            state + 0.5 * dt * k1,
            thrust,
            torque,
        )

        k3 = self.__state_derivative(
            state + 0.5 * dt * k2,
            thrust,
            torque,
        )

        k4 = self.__state_derivative(
            state + dt * k3,
            thrust,
            torque,
        )

        state_next = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        self.r = state_next[0:3]
        self.v = state_next[3:6]
        self.q = quat_normalize(state_next[6:10])
        self.omega = state_next[10:13]

        return state_next

    def state_from_flat_out(self, pos, vel, jerk, snap, yaw, yaw_omega, yaw_acc):
        """
        Compute state from flat output values

        Returns:
        - state: np.array - (r1,r2,r3,v1,v2,v3,q1,q2,q3,q4,omega1,omega2,omega3)
        """

        q = np.zeros((4,))  # TODO Policzyć
        omega = np.array([0.0, 0.0, yaw_omega])  # TODO: Policzyć

        return np.concatenate([pos, vel, q, omega])
