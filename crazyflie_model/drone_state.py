import numpy as np
import rclpy
from geometry_msgs.msg import Pose, Wrench
from nav_msgs.msg import Odometry
from rclpy.node import Node

from crazyflie_model.config import *
from crazyflie_model.drone import Drone
from crazyflie_model.utils import quaternion_to_euler


class CrazyflieModelNode(Node):
    def __init__(self):
        super().__init__('crazyflie_model')

        self.drone = Drone()
        self.dt = 0.01

        # Bezpośrednie sterowanie za pomocą thrust i torque
        self._input_sub = self.create_subscription(
            Wrench, 'crazyflie/model_inputs', self.input_callback, 10
        )
        # Example message (Hover):
        # ros2 topic pub /crazyflie/model_inputs geometry_msgs/msg/Wrench "{force: {x: 0.0, y: 0.0, z: 0.27468}, torque: {x: 0.0, y: 0.0, z: 0.0}}"

        # Sterowanie za pomocą zadanej pozycji i prędkości
        self._target_sub = self.create_subscription(
            Pose, 'crazyflie/target_pose', self.target_callback, 10
        )
        # Example message:
        # ros2 topic pub /crazyflie/target_pose geometry_msgs/msg/PoseStamped "{pose: {position: {x: 2.0, y: 2.0, z: 2.0}}}"

        # Tryb sterowania: True = używamy Mellingera, False = używamy Wrench
        self.use_controller = True

        # Aktualne sterowanie (inputs)
        self.thrust = G * MASS  # to hover in initial state
        self.torque = np.zeros(3)  # to hover in initial state

        self.target_pos = np.array([0.0, 0.0, 1.0])  # Domyślnie zawis na 1m
        self.target_vel = np.array([0.0, 0.0, 0.0])
        self.target_yaw = 0.0

        self.publisher = self.create_publisher(Odometry, 'crazyflie/state', 10)

        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info('Crazyflie model node started.')

    def input_callback(self, msg: Wrench):
        """Aktualizacja wejść sterujących dronem"""
        self.use_controller = False
        self.thrust = msg.force.z
        self.torque = np.array([msg.torque.x, msg.torque.y, msg.torque.z])

    def target_callback(self, msg: Pose):
        """Aktualizacja punktu docelowego"""
        self.use_controller = True
        self.target_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.target_yaw = quaternion_to_euler(
            (msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)
        )[2]

    def timer_callback(self):
        """Integracja stanu drona i publikacja co dt"""
        if self.use_controller:
            curr_state = self.drone.curr_state()

            # Mellinger
            self.thrust, self.torque = self.drone.mellinger_control(
                curr_state=curr_state,
                pos_target=self.target_pos,
                vel_target=self.target_vel,
                yaw_target=self.target_yaw,
                acc_target=np.array([0.0, 0.0, 0.0]),  # Celujemy w zawis/stałą prędkość
                omega_target=np.array([0.0, 0.0, 0.0]),
                k_p=4.5,
                k_v=3.5,
                k_R=0.5,
                k_omega=0.1,
            )

        # Appply control
        state = self.drone.state_model(self.thrust, self.torque, self.dt)

        # self.get_logger().info(f'State: {state}')
        self.publish_state(state)

    def publish_state(self, state):
        msg = Odometry()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.child_frame_id = 'crazyflie'

        # position
        msg.pose.pose.position.x = state[0]
        msg.pose.pose.position.y = state[1]
        msg.pose.pose.position.z = state[2]

        # velocity
        msg.twist.twist.linear.x = state[3]
        msg.twist.twist.linear.y = state[4]
        msg.twist.twist.linear.z = state[5]

        # quaternion
        msg.pose.pose.orientation.w = state[6]
        msg.pose.pose.orientation.x = state[7]
        msg.pose.pose.orientation.y = state[8]
        msg.pose.pose.orientation.z = state[9]

        # angular velocity
        msg.twist.twist.angular.x = state[10]
        msg.twist.twist.angular.y = state[11]
        msg.twist.twist.angular.z = state[12]

        self.publisher.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = CrazyflieModelNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
