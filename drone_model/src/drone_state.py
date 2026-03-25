import numpy as np
import rclpy
from geometry_msgs.msg import Wrench
from nav_msgs.msg import Odometry
from rclpy.node import Node

from drone_model import Drone


class CrazyflieModelNode(Node):
    def __init__(self):
        super().__init__('crazyflie_model')

        self.drone = Drone()

        self.dt = 0.01

        self.subscription = self.create_subscription(
            Wrench, 'crazyflie/model_inputs', self.input_callback, 10
        )

        self.publisher = self.create_publisher(Odometry, 'crazyflie/state', 10)

        self.get_logger().info('Crazyflie model node started.')

    def input_callback(self, msg: Wrench):
        thrust = msg.force.z
        torque = np.array([msg.torque.x, msg.torque.y, msg.torque.z])

        state = self.drone.forward(thrust, torque, self.dt)
        self.publish_state(state)

    def publish_state(self, state):

        msg = Odometry()

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

