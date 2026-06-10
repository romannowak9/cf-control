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

        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])

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
        m = self.mass
        zw = np.array([0.0, 0.0, 1.0])

        # 1. Wektor ciągu, jego moduł i oś z_B
        t = acc + self.gravity * zw
        norm_t = np.linalg.norm(t)
        thrust = m * norm_t

        if norm_t < 1e-6:
            zb = zw
        else:
            zb = t / norm_t

        # 2. Wyznaczenie osi x_B oraz y_B
        xc = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        yb_cross_tmp = np.cross(zb, xc)

        if np.linalg.norm(yb_cross_tmp) < 1e-6:
            yb = np.array([0.0, 1.0, 0.0])
        else:
            yb = yb_cross_tmp / np.linalg.norm(yb_cross_tmp)

        xb = np.cross(yb, zb)

        # Obliczenie kwaternionu z macierzy rotacji
        rotation = np.column_stack((xb, yb, zb))
        q = rotation_matrix_to_quaternion(rotation)
        q = quat_normalize(q)

        # 3. Pierwsza pochodna ciągu i prędkości kątowe (wykorzystując Jerk)
        u1_dot = m * np.dot(jerk, zb)
        h_omega = (m * jerk - u1_dot * zb) / thrust

        omega_x = -np.dot(h_omega, yb)
        omega_y = np.dot(h_omega, xb)
        omega_z = yaw_dot * np.dot(zw, zb)

        omega = np.array([omega_x, omega_y, omega_z])

        # 4. Druga pochodna ciągu i przyspieszenia kątowe (wykorzystując Snap)
        # Wykorzystujemy u1_ddot = m * (s o zb) + thrust * ||h_omega||^2
        u1_ddot = m * np.dot(snap, zb) + thrust * np.dot(h_omega, h_omega)

        # Wektor h_alpha
        h_alpha = (
            m * snap - u1_ddot * zb - 2 * u1_dot * h_omega - thrust * np.cross(omega, h_omega)
        ) / thrust

        # Rzutowanie na osie lokalne drona
        omega_dot_x = -np.dot(h_alpha, yb)
        omega_dot_y = np.dot(h_alpha, xb)
        omega_dot_z = yaw_acc * np.dot(zw, zb)

        omega_dot = np.array([omega_dot_x, omega_dot_y, omega_dot_z])

        # 5. Wyliczenie momentów sił (Torque Feedforward)
        torque = self.J @ omega_dot + np.cross(omega, self.J @ omega)

        return np.concatenate([pos, vel, q, omega]), np.concatenate([[thrust], torque]), omega_dot

    def mellinger_control(
        self,
        curr_state,
        ref_state,
        ref_thrust,
        ref_alpha,
        yaw_ref,
        k_p=1,
        k_v=1,
        k_R=1,
        k_omega=1,
    ):
        pos_curr = curr_state[:3]
        vel_curr = curr_state[3:6]
        q_curr = curr_state[6:10]
        omega_curr = curr_state[10:13]

        pos_ref = ref_state[:3]
        vel_ref = ref_state[3:6]
        q_ref = ref_state[6:10]
        omega_ref = ref_state[10:13]

        R_ref = quaternion_to_rotation_matrix(q_ref)
        R_curr = quaternion_to_rotation_matrix(q_curr)

        Kp = np.eye(3) * k_p
        Kv = np.eye(3) * k_v
        KR = np.eye(3) * k_R
        Komega = np.eye(3) * k_omega

        zw = np.array([0, 0, 1])

        # Position / velocity errors
        error_pos = pos_curr - pos_ref
        error_vel = vel_curr - vel_ref

        # Reference acceleration
        acc_ref = ref_thrust / self.mass * R_ref[:, 2] - self.gravity * zw

        # Desired force
        F_des = (
            -Kp @ error_pos - Kv @ error_vel + self.mass * self.gravity * zw + self.mass * acc_ref
        )

        norm_F = np.linalg.norm(F_des)

        if norm_F < 1e-6:
            zb_des = np.array([0.0, 0.0, 1.0])
        else:
            zb_des = F_des / norm_F

        xc_des = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])

        yb_des = np.cross(zb_des, xc_des)

        if np.linalg.norm(yb_des) < 1e-6:
            yb_des = np.cross(zb_des, R_ref[:, 1])

        yb_des /= np.linalg.norm(yb_des)

        xb_des = np.cross(yb_des, zb_des)

        R_des = np.column_stack((xb_des, yb_des, zb_des))

        zb = R_curr[:, 2]

        thrust = float(F_des @ zb)

        error_R = vee(0.5 * (R_des.T @ R_curr - R_curr.T @ R_des))

        error_omega = omega_curr - R_curr.T @ R_des @ omega_ref

        ref_alpha_body = R_curr.T @ R_des @ ref_alpha  # Obrót do układu lokalnego

        torque = (
            -KR @ error_R
            - Komega @ error_omega
            + np.cross(omega_curr, self.J @ omega_curr)
            + self.J @ ref_alpha_body
        )

        if not np.all(np.isfinite(torque)):
            torque = np.zeros(3)

        thrust = np.clip(thrust, 0.0, 4.5)
        torque = np.clip(torque, -0.05, 0.05)

        return thrust, torque

    @staticmethod
    def generate_reference_trajectory(t_eval, T_flight, p0, pT, yaw0, yawT):
        """
        Generuje optymalną (minimum snap) trajektorię punkt-punkt w czasie t_eval.

        Parametry:
        - t_eval: aktualny czas od rozpoczęcia manewru (sekundy)
        - T_flight: całkowity zaplanowany czas na wykonanie manewru (sekundy)
        - p0, pT: pozycje startowa i docelowa (wektory 3D)
        - yaw0, yawT: yaw startowy i docelowy (radiany)
        """
        # Ogranicz czas do maksymalnego czasu lotu, aby dron zatrzymał się u celu
        t = np.clip(t_eval, 0.0, T_flight)

        # Czas znormalizowany tau w przedziale [0, 1]
        tau = t / T_flight

        if t >= T_flight:
            # Gdy dolecieliśmy, utrzymuj pozycję (zawis)
            return pT, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), yawT, 0.0, 0.0

        # Współczynniki znormalizowanego wielomianu minimum snap 7. stopnia
        # Wymusza zerową prędkość, przyspieszenie i szarpnięcie na początku (tau=0) i na końcu (tau=1)
        c4, c5, c6, c7 = 35.0, -84.0, 70.0, -20.0

        # Ewaluacja wielomianu i jego pochodnych po tau
        p_tau = c4 * tau**4 + c5 * tau**5 + c6 * tau**6 + c7 * tau**7
        dp_tau = 4 * c4 * tau**3 + 5 * c5 * tau**4 + 6 * c6 * tau**5 + 7 * c7 * tau**6
        ddp_tau = 12 * c4 * tau**2 + 20 * c5 * tau**3 + 30 * c6 * tau**4 + 42 * c7 * tau**5
        dddp_tau = 24 * c4 * tau + 60 * c5 * tau**2 + 120 * c6 * tau**3 + 210 * c7 * tau**4
        ddddp_tau = 24 * c4 + 120 * c5 * tau + 360 * c6 * tau**2 + 840 * c7 * tau**3

        # Transformacja przestrzenna (Spatial Scaling) i czasowa (Temporal Scaling) z artykułu Mellingera
        delta_p = pT - p0
        pos = p0 + delta_p * p_tau
        vel = delta_p * dp_tau / T_flight
        acc = delta_p * ddp_tau / (T_flight**2)
        jerk = delta_p * dddp_tau / (T_flight**3)
        snap = delta_p * ddddp_tau / (T_flight**4)

        # Artykuł wspomina k_psi = 2 dla yaw (czyli minimum yaw acceleration), ale dla gładkości
        # i wygody użyjemy tej samej kinematyki 7. stopnia co dla pozycji.
        delta_yaw = yawT - yaw0
        yaw = yaw0 + delta_yaw * p_tau
        yaw_dot = delta_yaw * dp_tau / T_flight
        yaw_acc = delta_yaw * ddp_tau / (T_flight**2)

        return pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_acc
