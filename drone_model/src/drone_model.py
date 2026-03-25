import numpy as np
from config import *


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
    def __init__(self, mass=MASS, inertia_matrix=J, init_pose=INIT_POSE, init_vel=INIT_VEL, g=G):

        self.mass = mass
        self.J = inertia_matrix
        self.J_inv = np.linalg.inv(self.J)

        # Initial state
        self.r = np.array(init_pose[:3], dtype=float)
        self.v = np.array(init_vel[:3], dtype=float)
        self.q = euler_to_quaternion(*init_pose[3:])
        self.omega = np.array(init_vel[3:], dtype=float)

        self.g = g

    def __state_derivative(self, thrust, torque, dt):
        r_dot = self.v

        thrust_body = np.array([0, 0, thrust])
        thrust_world = quat_rotate(self.q, thrust_body)

        v_dot = -self.g + (1 / self.mass) * thrust_world

        omega_quat = np.array([0, self.omega[0] / 2, self.omega[1] / 2, self.omega[2] / 2])

        q_dot = quat_multiply(self.q, omega_quat)

        omega_dot = self.J_inv @ (torque - np.cross(self.omega, self.J @ self.omega))

        self.r += r_dot * dt
        self.v += v_dot * dt

        self.q += q_dot * dt
        self.q = quat_normalize(self.q)

        self.omega += omega_dot * dt

        return np.concatenate([r_dot, v_dot, q_dot, omega_dot])

    def forward(self, thrust, torque, dt):
        """
        Compute state from input values

        Returns:
        - state: np.array - (r1,r2,r3,v1,v2,v3,q1,q2,q3,q4,omega1,omega2,omega3,omega4)
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

