import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from rclpy.node import Node

from cf_control_msgs.msg import ThrustAndTorque
from crazyflie_model.config import *
from crazyflie_model.drone import Drone
from crazyflie_model.utils import quaternion_to_euler


class CrazyflieModelNode(Node):
    def __init__(self):
        super().__init__('crazyflie_model')

        self.drone = Drone()
        self.dt = 0.002

        # 1. Subskrypcja punktu docelowego (celu)
        self._target_sub = self.create_subscription(
            Pose, 'crazyflie/target_pose', self.target_callback, 10
        )
        # Przykładowa wiadomość
        # ros2 topic pub --once /crazyflie/target_pose geometry_msgs/msg/Pose "{position: {x: 0.0, y: 0.0, z: 1.5}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}"

        # 2. Subskrypcja rzeczywistej odometrii drona (sprzężenie zwrotne)
        self._odom_sub = self.create_subscription(
            Odometry, '/crazyflie/odom', self.odom_callback, 10
        )

        # Aktualne sterowanie (inputs)
        self.thrust = G * MASS  # to hover in initial state
        self.torque = np.zeros(3)  # to hover in initial state

        self.target_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.target_pos = np.array([0.0, 0.0, 1.0])  # Domyślnie zawis na 1m
        self.target_vel = np.array([0.0, 0.0, 0.0])
        self.target_yaw = 0.0

        # Zmienne do przechowywania aktualnego stanu z odometrii
        # Wektor 13-elementowy: [pos(3), vel(3), quat(4), omega(3)]
        self.curr_state = np.zeros(13)
        self.state_received = False  # Flaga bezpieczeństwa

        # Bezpośrednie sterowanie za pomocą thrust i torque
        self._input_pub = self.create_publisher(ThrustAndTorque, '/cf_control/control_command', 10)

        # Timer pętli sterowania (częstotliwość 500Hz)
        self.timer = self.create_timer(self.dt, self.timer_callback)

        # Zmienne zarządzania czasem trajektorii
        self.t_start = None  # Czas odebrania komendy (ros2 time)
        self.T_flight = 3.0  # Zadeklarowany czas przelotu (np. 3 sekundy)

        # Zmienne początkowe trajektorii
        self.start_pos = np.array([0.0, 0.0, 0.0])
        self.start_yaw = 0.0

        self.get_logger().info('Crazyflie controller node started.')

    def target_callback(self, msg: Pose):
        """Aktualizacja punktu docelowego i uruchomienie trajektorii"""

        # Nie przypisujemy celu, jeśli nie znamy własnej pozycji z odometrii
        if not self.state_received:
            self.get_logger().warn('Cannot set target: No odometry received yet.')
            return

        # Ustawienie punktu początkowego jako obecnej pozycji z odometrii
        self.start_pos = self.curr_state[:3].copy()
        self.start_yaw = quaternion_to_euler(self.curr_state[6:10])[2]

        # Ustawienie celu
        self.target_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        target_q = np.array(
            [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        )
        self.target_yaw = quaternion_to_euler(target_q)[2]

        # Resetowanie czasu - zaczynamy generację trajektorii od t=0
        self.t_start = self.get_clock().now()

        self.get_logger().info(
            f'New target received! Planning {self.T_flight}s minimum-snap flight.'
        )

    def odom_callback(self, msg: Odometry):
        """Aktualizacja rzeczywistego stanu drona z zewnętrznego źródła"""
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

        # Łączymy w jeden wektor stanu kompatybilny z Twoim kontrolerem
        self.curr_state = np.concatenate([pos, vel, quat, omega])
        self.state_received = True

    def timer_callback(self):
        """Full Mellinger pipeline (Czysty Kontroler)"""

        if not self.state_received:
            self.get_logger().warn('Waiting for odometry...', throttle_duration_sec=2.0)
            return

        curr_state = self.curr_state

        # Oblicz, ile czasu upłynęło od odebrania targetu
        if self.t_start is None:
            # Jeśli nie podano jeszcze celu, zachowaj aktualną pozycję (t_eval = T_flight aby wymusić zawis w miejscu startu)
            t_eval = self.T_flight
            self.start_pos = curr_state[:3]
            self.target_pos = curr_state[:3]
            self.start_yaw = quaternion_to_euler(curr_state[6:10])[2]
            self.target_yaw = self.start_yaw
        else:
            t_eval = (self.get_clock().now() - self.t_start).nanoseconds / 1e9

        # 1. Wygeneruj flat outputs dla obecnej milisekundy t_eval
        (pos_ref, vel_ref, acc_ref, jerk_ref, snap_ref, yaw_ref, yaw_dot_ref, yaw_acc_ref) = (
            Drone.generate_reference_trajectory(
                t_eval=t_eval,
                T_flight=self.T_flight,
                p0=self.start_pos,
                pT=self.target_pos,
                yaw0=self.start_yaw,
                yawT=self.target_yaw,
            )
        )

        # 2. Differential flatness (wykorzystuje Twoją zmodyfikowaną funkcję z poprawką B)
        ref_state, ref_control, ref_alpha = self.drone.flat_out_state_and_control(
            pos_ref, vel_ref, acc_ref, jerk_ref, snap_ref, yaw_ref, yaw_dot_ref, yaw_acc_ref
        )

        # 3. Mellinger controller
        self.thrust, self.torque = self.drone.mellinger_control(
            curr_state=curr_state,
            ref_state=ref_state,
            ref_thrust=ref_control[0],
            ref_alpha=ref_alpha,
            yaw_ref=yaw_ref,
            k_p=K_P,
            k_v=K_V,
            k_R=K_R,
            k_omega=K_OMEGA,
        )

        # 4. Publikacja wyliczonego sterowania
        self.publish_drone_input(self.thrust, self.torque)

    def publish_drone_input(self, thrust, torque):
        msg = ThrustAndTorque()
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
