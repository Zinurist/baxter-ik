# Baxter Inverse Kinematics using optimization
This project provides an interface to calculate forward/inverse kinematics for the baxter robot using numpy and the scipy optimization library.

The `transformations.py` file from the ROS Tf python package is included and needed. If you want to use your own file, you can replace the import at the top of `baxter_ik.py`.

### Forward Kinematics
This project offers functions for calculating forward kinematics as a numpy matrix, using tf's `tranformations.py`. 
As reference for the forward kinematics, the urdf files from the [baxter_common](https://github.com/RethinkRobotics/baxter_common) repository where used, specifically the [baxter description](https://github.com/RethinkRobotics/baxter_common/blob/master/baxter_description/urdf/baxter.urdf) and the [gripper description](https://github.com/RethinkRobotics/baxter_common/blob/master/rethink_ee_description/urdf/electric_gripper/rethink_electric_gripper.xacro).

For more information on the forward kinematics, look at the documentation of the `forward_matrix` function in `baxter_ik.py`.

### Inverse Kinematics
Here, the inverse kinematics problems is formulated as minimizing a loss function with the joint values as parameters. This definition allows us to create custom goals very easily. 
See [example.py](example.py) for an example using a custom pointing loss. With this loss function, we can find the best/closest pose (for either robot arm) for pointing at some position.

You can also use the scipy `minimize` function yourself, look at the `baxter_ik` function in `baxter_ik.py` for an example.
You can then for example add your own constraints to the optimization problem.

Note that this IK doesn't do any collision detection. It does however consider the joint limits of the robot.

