# Third-party notice: Autoware MPC lateral controller (kinematics)

This repository contains a **pure-Python port** of Autoware Universe’s MPC lateral controller
(`autoware_mpc_lateral_controller`) for use as a baseline path-tracking controller.

## Upstream
- Upstream repository: `autowarefoundation/autoware_universe`

## Upstream source files referenced for the port
- `control/autoware_mpc_lateral_controller/src/mpc.cpp`
- `control/autoware_mpc_lateral_controller/include/autoware/mpc_lateral_controller/mpc.hpp`
- `control/autoware_mpc_lateral_controller/src/vehicle_model/vehicle_model_bicycle_kinematics.cpp`
- `control/autoware_mpc_lateral_controller/src/mpc_utils.cpp`
- `control/autoware_mpc_lateral_controller/src/lowpass_filter.cpp`
- `control/autoware_mpc_lateral_controller/param/lateral_controller_defaults.param.yaml`

## License
Autoware Universe is licensed under the **Apache License, Version 2.0**.
The Python port in this repository is derived from the files listed above and is provided under
the same license terms for those derived portions.

Apache-2.0 license text: `http://www.apache.org/licenses/LICENSE-2.0`

## Notes about this port
- This is **not** the ROS2 node implementation; it does not use ROS messages, rclcpp, or Autoware
  build system components.
- The port implements the **kinematics bicycle** MPC (state: lateral error, yaw error, steer;
  input: steer command) and uses **OSQP** (via python `osqp`) to solve the QP.
