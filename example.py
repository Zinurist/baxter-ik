from baxter_ik import *

# This example uses the pointing loss for the optimization.
# When moving the robot to the resulting joint values,
# he should be roughly pointing at the goal position with his right gripper.

object_position = np.array([1.294, -0.063, 0.003])
joints = baxter_ik(create_pointing_loss(object_position, limb='right'))
joints_dict = convert_joints_to_dict(joints, 'right')
print('To point at object at position %s, move joints to positions %s.' % (object_position, joints_dict))


joints = baxter_ik(create_position_loss(object_position, limb='right'))
joints_dict = convert_joints_to_dict(joints, 'right')
print('To get to position %s as close as possible, move joints to positions %s.' % (object_position, joints_dict))