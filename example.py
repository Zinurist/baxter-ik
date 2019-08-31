from baxter_ik import *

# This example uses the pointing loss for the optimization.
# When moving the robot to the resulting joint values,
# he should be roughly pointing at the goal position with his right gripper.

object_position = np.array([1.294, -0.063, 0.003])
res = baxter_ik(create_pointing_loss(object_position, limb='right', fix_joints=np.array([False, False, True, False, True, False, False])))
print(res)
joints_dict = convert_joints_to_dict(res.x, 'right')
print('To point at object at position %s, move joints to positions %s. Loss: %s' % (object_position, joints_dict, res.fun))


res = baxter_ik(create_position_loss(object_position, limb='right'))
print(res)
joints_dict = convert_joints_to_dict(res.x, 'right')
print('To get to position %s as close as possible, move joints to positions %s. Loss: %s' % (object_position, joints_dict, res.fun))
    