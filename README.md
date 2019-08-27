This project provides an interface to calculate inverse kinematics for the baxter robot using the scipy optimization library.

Here, the inverse kinematics problems is formulated as minimizing a loss function with the joint values as parameters. This definition allows us to create custom goals very easily.

See [example.py](example.py) for an example using a custom pointing loss. With this loss function, we can find the best/closest pose (for either robot arm) for pointing at some position.