import numpy as np

from crazyflie_model.config import *
from crazyflie_model.utils import (
    euler_to_quaternion,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quaternion_to_rotation_matrix,
    rk4,
    rotation_matrix_to_quaternion,
    vee,
)


class Drone:
    def __init__(
        self, mass=MASS, inertia_matrix=J, init_pose=INIT_POSE, init_vel=INIT_VEL, gravity=G
    ):

        self.mass = mass
        self.J = inertia_matrix
        self.J_inv = np.linalg.inv(self.J)
        self.gravity = gravity

        # Initial state
        self.r = np.array(init_pose[:3], dtype=float)
        self.v = np.array(init_vel[:3], dtype=float)
        self.q = quat_normalize(euler_to_quaternion(*init_pose[3:]))
        self.omega = np.array(init_vel[3:], dtype=float)

    def curr_state(self):
        return np.concatenate([self.r, self.v, self.q, self.omega])

    def __state_derivative(self, state, thrust, torque):
        v = state[3:6]
        q = state[6:10]
        omega = state[10:13]

        r_dot = v

        thrust_body = np.array([0.0, 0.0, thrust])
        thrust_world = quat_rotate(q, thrust_body)

        v_dot = np.array([0.0, 0.0, -self.gravity]) + (1 / self.mass) * thrust_world

        omega_quat = np.array([0.0, omega[0] / 2.0, omega[1] / 2.0, omega[2] / 2.0])

        q_dot = 0.5 * quat_multiply(q, omega_quat)

        omega_dot = self.J_inv @ (torque - np.cross(omega, self.J @ omega))

        return np.concatenate([r_dot, v_dot, q_dot, omega_dot])

    def state_model(self, thrust, torque, dt):
        """
        Compute state from input values

        Returns:
        - state: np.array - (p1,p2,p3,v1,v2,v3,q1,q2,q3,q4,omega1,omega2,omega3)
        """

        state = np.concatenate([self.r, self.v, self.q, self.omega])

        state_next = rk4(self.__state_derivative, state, dt, thrust=thrust, torque=torque)

        self.r = state_next[0:3]
        self.v = state_next[3:6]
        self.q = quat_normalize(state_next[6:10])
        if not np.all(np.isfinite(self.q)):
            self.q = np.array([1, 0, 0, 0])
        self.omega = np.clip(state_next[10:13], -50.0, 50.0)

        return state_next

    def flat_out_state_and_control(self, pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_acc):
        """
        Compute state from flat output values

        Returns:
        - state and control : Tuple
            (
                - state: np.array - (p1,p2,p3,v1,v2,v3,q1,q2,q3,q4,omega1,omega2,omega3)
                - control: np.array - (thrust, torque_x, torque_y, torque_z)
            )
        """

        a_g = acc + np.array([0, 0, self.gravity])
        thrust = self.mass * np.linalg.norm(a_g)
        zb = a_g / np.linalg.norm(a_g)
        xc = np.array([np.cos(yaw), np.sin(yaw), 0])
        yb_cross_tmp = np.cross(zb, xc)
        yb = yb_cross_tmp / np.linalg.norm(yb_cross_tmp)
        xb = np.cross(yb, zb)

        rotation = np.column_stack((xb, yb, zb))
        q = rotation_matrix_to_quaternion(rotation)
        q = quat_normalize(q)

        h_omega = (self.mass / thrust) * (jerk - (jerk @ zb) * zb)

        omega_x = -h_omega @ yb
        omega_y = h_omega @ xb
        zw = np.array([0, 0, 1])
        omega_z = yaw_dot * (zw @ zb)

        omega = np.array([omega_x, omega_y, omega_z])

        omega_dot_x = -(
            (self.mass / thrust) * snap[1]
            - 2 * (self.mass / thrust) * jerk[2] * omega_x
            - omega_y * omega_z
        )

        omega_dot_y = (
            (self.mass / thrust) * snap[0]
            - 2 * (self.mass / thrust) * jerk[2] * omega_y
            - omega_x * omega_z
        )

        omega_dot_z = yaw_acc * (zw @ zb)

        omega_dot = np.array([omega_dot_x, omega_dot_y, omega_dot_z])

        torque = self.J @ omega_dot + np.cross(omega, self.J @ omega)

        return np.concatenate([pos, vel, q, omega]), np.concatenate([[thrust], torque])

    def mellinger_control(
        self,
        curr_state,
        pos_target,
        vel_target,
        yaw_target,
        acc_target,
        omega_target,
        k_p=1,
        k_v=1,
        k_R=1,
        k_omega=1,
    ):
        pos_curr = curr_state[:3]
        vel_curr = curr_state[3:6]
        q_curr = curr_state[6:10]
        omega_curr = curr_state[10:13]

        # Sekcja control z papera o Melingerze
        error_pos = pos_curr - pos_target
        error_vel = vel_curr - vel_target

        Kp = np.eye(3) * k_p
        Kv = np.eye(3) * k_v
        KR = np.eye(3) * k_R
        Komega = np.eye(3) * k_omega

        zw = np.array([0, 0, 1])

        F_des = (
            (-Kp) @ error_pos
            - Kv @ error_vel
            + self.mass * self.gravity * zw
            + self.mass * acc_target
        )

        norm_F = np.linalg.norm(F_des)
        if norm_F < 1e-6:
            return float(self.mass * self.gravity), np.zeros(3)

        zb_des = F_des / norm_F
        xc_des = np.array([np.cos(yaw_target), np.sin(yaw_target), 0])
        yb_des_cross_temp = np.cross(zb_des, xc_des)

        norm_y = np.linalg.norm(yb_des_cross_temp)
        if norm_y < 1e-6:
            yb_des = np.array([0.0, 1.0, 0.0])
        else:
            yb_des = yb_des_cross_temp / norm_y

        xb_des = np.cross(yb_des, zb_des)

        R_des = np.column_stack((xb_des, yb_des, zb_des))

        R_curr = quaternion_to_rotation_matrix(q_curr)
        zb = R_curr[:, 2]

        # thrust = float(np.clip(F_des @ zb, 0.0, 2.0 * self.mass * self.gravity))
        thrust = float(np.linalg.norm(F_des))

        error_R = vee(0.5 * (R_des.T @ R_curr - R_curr.T @ R_des))
        error_omega = omega_curr - R_curr.T @ R_des @ omega_target

        torque = -KR @ error_R - Komega @ error_omega
        torque = np.clip(torque, -1e-3, 1e-3)

        if not np.all(np.isfinite(torque)):
            torque = np.zeros(3)

        print(torque, thrust)

        return thrust, torque
