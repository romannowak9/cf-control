import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from rclpy.node import Node

from cf_control_msgs.msg import ThrustAndTorque
from crazyflie_model.config import *
from crazyflie_model.drone import Drone
from crazyflie_model.mellinger_trajectory import MellingerTrajectory
from crazyflie_model.utils import quaternion_to_euler


class CrazyflieModelNode(Node):
    def __init__(self):
        super().__init__('crazyflie_model')

        self.drone = Drone()
        self.dt = 0.002  # 500Hz

        # Subskrypcje
        self._target_sub = self.create_subscription(
            Pose, 'crazyflie/target_pose', self.target_callback, 10
        )
        self._odom_sub = self.create_subscription(
            Odometry, '/crazyflie/odom', self.odom_callback, 10
        )

        # Publikacja sterowania
        self._input_pub = self.create_publisher(ThrustAndTorque, '/cf_control/control_command', 10)

        # Zmienne stanu i bezpieczeństwa
        self.curr_state = np.zeros(13)
        self.state_received = False

        # Zmienne zarządzania czasem (Zsynchronizowane z Gazebo)
        self.current_sim_time = 0.0
        self.t_start = None
        self.T_flight = 3.0  # Czas przelotu segmentu
        self.active_trajectory = None

        # Pamięć ostatniego celu (Zabezpieczenie przed spamem wiadomości)
        self.last_target_pos = None
        self.last_target_yaw = None

        # Timer pętli sterowania (500Hz)
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info('Crazyflie Mellinger controller node for Gazebo started.')

    def odom_callback(self, msg: Odometry):
        """Aktualizacja rzeczywistego stanu drona oraz czasu z symulatora Gazebo"""
        # POPRAWKA 1: Pobieranie czasu bezpośrednio z fizyki Gazebo (msg.header.stamp)
        self.current_sim_time = msg.header.stamp.sec + (msg.header.stamp.nanosec / 1e9)

        pos = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        )
        vel = np.array(
            [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        )
        quat = np.array(
            [
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
            ]
        )
        omega = np.array(
            [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        )

        self.curr_state = np.concatenate([pos, vel, quat, omega])

        # Inicjalizacja stabilnego zawisu w punkcie startowym przy pierwszym uruchomieniu
        if not self.state_received:
            init_yaw = quaternion_to_euler(quat)[2]
            self.active_trajectory = MellingerTrajectory(
                pos, pos, init_yaw, init_yaw, self.T_flight
            )
            self.state_received = True

    def target_callback(self, msg: Pose):
        """Aktualizacja punktu docelowego i wygenerowanie nowej trajektorii"""
        if not self.state_received:
            self.get_logger().warn('Cannot set target: No odometry received yet.')
            return

        # Pobranie parametrów celu z wiadomości ROS
        target_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        target_q = np.array(
            [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        )
        target_yaw = quaternion_to_euler(target_q)[2]

        # POPRAWKA 2: Ignorowanie identycznych punktów, aby ciągła publikacja nie resetowała czasu
        if self.last_target_pos is not None:
            pos_identical = np.allclose(target_pos, self.last_target_pos, atol=1e-3)
            yaw_identical = np.isclose(target_yaw, self.last_target_yaw, atol=1e-3)
            if pos_identical and yaw_identical:
                return  # Cel się nie zmienił, kontynuuj obecny lot

        # Zapisanie nowego celu do pamięci podręcznej
        self.last_target_pos = target_pos
        self.last_target_yaw = target_yaw

        # Początek trajektorii to aktualna pozycja drona w przestrzeni
        start_pos = self.curr_state[:3].copy()
        start_yaw = quaternion_to_euler(self.curr_state[6:10])[2]

        # Utworzenie nowej instancji trajektorii Minimum Snap
        self.active_trajectory = MellingerTrajectory(
            start_pos, target_pos, start_yaw, target_yaw, self.T_flight
        )

        # POPRAWKA 3: Zapisanie czasu startu na podstawie czasu symulacji Gazebo
        self.t_start = self.current_sim_time
        self.get_logger().info(
            f'New target received! Target: {target_pos}. Planning {self.T_flight}s flight.'
        )

    def timer_callback(self):
        """Pętla sterowania pracująca z częstotliwością 500Hz"""
        if not self.state_received:
            self.get_logger().warn('Waiting for odometry...', throttle_duration_sec=2.0)
            return

        # POPRAWKA 4: Obliczenie czasu trwania lotu na podstawie osi czasu Gazebo
        if self.t_start is None:
            t_eval = 0.0  # Ewaluacja bezpiecznego zawisu początkowego
        else:
            t_eval = self.current_sim_time - self.t_start

        # POPRAWKA 5: Bezpieczne ograniczenie czasu ewaluacji do czasu trwania lotu.
        # Zapobiega to szarpnięciom, jeśli klasa MellingerTrajectory ma niestabilny warunek brzegowy 'if t_eval >= T_flight'
        t_eval_safe = min(t_eval, self.T_flight)

        # 1. Pobranie wartości Flat Outputs z obiektu trajektorii
        (pos_ref, vel_ref, acc_ref, jerk_ref, snap_ref, yaw_ref, yaw_dot_ref, yaw_acc_ref) = (
            self.active_trajectory.evaluate(t_eval_safe)
        )

        # 2. Mapowanie Flat Outputs na żądany stan i sterowanie referencyjne
        ref_state, ref_control, ref_alpha = self.drone.flat_out_state_and_control(
            pos_ref, vel_ref, acc_ref, jerk_ref, snap_ref, yaw_ref, yaw_dot_ref, yaw_acc_ref
        )

        # 3. Kontroler geometryczny Mellingera (Wyznaczenie wypadkowego Thrust i Torque)
        thrust, torque = self.drone.mellinger_control(
            curr_state=self.curr_state,
            ref_state=ref_state,
            ref_thrust=ref_control[0],
            ref_alpha=ref_alpha,
            yaw_ref=yaw_ref,
            k_p=K_P,
            k_v=K_V,
            k_R=K_R,
            k_omega=K_OMEGA,
        )

        # 4. Publikacja wyliczonego sterowania do drona
        self.publish_drone_input(thrust, torque)

    def publish_drone_input(self, thrust, torque):
        msg = ThrustAndTorque()
        # Znacznik czasu generowany zgodnie z zegarem węzła (który w symulacji używa sim_time)
        msg.timestamp = int(self.get_clock().now().nanoseconds)
        msg.collective_thrust = float(thrust)
        msg.torque.x = torque[0]
        msg.torque.y = torque[1]
        msg.torque.z = torque[2]
        self._input_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CrazyflieModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
