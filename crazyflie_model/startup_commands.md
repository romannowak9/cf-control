```bash
cd /home/developer/ros2_ws
source ./setup.sh 41
./build.sh
source install/setup.bash
```

First terminal:
```bash
ros2 launch ros_gz_crazyflie_bringup crazyflie_simulation.launch.py
```

Second terminal:
```bash
ros2 run cf_control mixer
```

Third terminal:
```bash
ros2 run crazyflie_model drone_state --ros-args -p use_sim_time:=true
```

Fourth terminal:
```bash
ros2 topic pub --once /crazyflie/target_pose geometry_msgs/msg/Pose "{position: {x: 4.0, y: 1.0, z: 2.0}, orientation: {w: 1.0, x: 0.0, y: 0.0, z: 0.0}}"
```
or
```bash
ros2 topic pub /crazyflie/target_pose geometry_msgs/msg/Pose "{position: {x: 4.0, y: 1.0, z: 2.0}, orientation: {w: 1.0, x: 0.0, y: 0.0, z: 0.0}}"
```