from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package='crazyflie_model',
                executable='drone_state',
                name='drone_state',
                output='screen',
                parameters=[{'use_sim_time': True}],
            ),
        ]
    )
