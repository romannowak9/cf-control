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

        # Sterowanie za pomocą zadanej pozycji i prędkości
        self._target_sub = self.create_subscription(
            Pose, 'crazyflie/target_pose', self.target_callback, 10
        )
        # Example coomand:
        # ros2 topic pub /crazyflie/target_pose geometry_msgs/msg/Pose "{position: {x: 5.0, y: 5.0, z: 5.0}, orientation: {w: 1.0, x: 0.0, y: 0.0, z: 0.0}}"

        # Aktualne sterowanie (inputs)
        self.thrust = G * MASS  # to hover in initial state
        self.torque = np.zeros(3)  # to hover in initial state

        self.target_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.target_pos = np.array([0.0, 0.0, 1.0])  # Domyślnie zawis na 1m
        self.target_vel = np.array([0.0, 0.0, 0.0])
        self.target_yaw = 0.0

        self._state_pub = self.create_publisher(Odometry, 'crazyflie/state', 10)
        # Bezpośrednie sterowanie za pomocą thrust i torque
        self._input_pub = self.create_publisher(ThrustAndTorque, '/cf_control/control_command', 10)
        # Example command (Hover):
        # ros2 topic pub /cf_control/control_command cf_control_msgs/msg/ThrustAndTorque "{collective_thrust: 0.295}"

        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info('Crazyflie model node started.')

    def target_callback(self, msg: Pose):
        """Aktualizacja punktu docelowego"""

        self.target_pos = np.array([msg.position.x, msg.position.y, msg.position.z])

        self.target_q = np.array(
            [
                msg.orientation.w,
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
            ]
        )

        self.target_yaw = quaternion_to_euler(self.target_q)[2]

    def timer_callback(self):
        """Full Mellinger pipeline"""

        curr_state = self.drone.curr_state()
        # 1. Generate flat outputs
        (
            pos_ref,
            vel_ref,
            acc_ref,
            jerk_ref,
            snap_ref,
            yaw_ref,
            yaw_dot_ref,
            yaw_acc_ref,
        ) = Drone.generate_reference_trajectory(
            self.target_pos, self.target_yaw
        )  # Na podstawie lokalizacji punktu i czasu dotarcia do niego generuję trajektorię, czyli oczekiwany stan w czasie (wektor wejść dla flat_out_state)

        # 2. Differential flatness
        ref_state, ref_control, ref_alpha = self.drone.flat_out_state_and_control(
            pos_ref,
            vel_ref,
            acc_ref,
            jerk_ref,
            snap_ref,
            yaw_ref,
            yaw_dot_ref,
            yaw_acc_ref,
        )

        # 3. Extract feedforward terms
        ref_thrust = ref_control[0]
        ref_omega = ref_state[10:13]

        # 4. Mellinger controller
        self.thrust, self.torque = self.drone.mellinger_control(
            curr_state=curr_state,
            ref_state=ref_state,
            ref_thrust=ref_thrust,
            ref_omega=ref_omega,  # po prostu omega z targewt state
            ref_alpha=ref_alpha,  # czyli omega_dot
            k_p=4.5,
            k_v=3.5,
            k_R=0.5,
            k_omega=0.1,
        )

        self.get_logger().info(f'thrust={self.thrust:.3f}, torque={self.torque}')

        # 5. Apply dynamics
        state = self.drone.state_model(self.thrust, self.torque, self.dt)

        self.publish_state(state)
        self.publish_drone_input(self.thrust, self.torque)

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

        self._state_pub.publish(msg)

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
