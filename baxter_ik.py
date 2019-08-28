"""
A simple interface that uses scipy's optimization package to do inverse kinematics for the baxter robot.
This allows for customs goals for IK, for example "pointing at something", see example.py.

The joint bounds are taken from the baxter specification (see http://sdk.rethinkrobotics.com/wiki/Hardware_Specifications), 
and the matrices for the forward kinematics are extracted from the urdf description of the robot (see https://github.com/RethinkRobotics/baxter_common).
"""

import numpy as np
from scipy.optimize import minimize

# Replace this import with your own transformations.py if you want.
from transformations import *


def convert_joints_to_array(joints):
    """Converts the joints dict used by the baxter sdk to a numpy array used for this IK algorithm.
    """
    if 'right_s0' in joints:
        return np.array([joints['right_s0'], joints['right_s1'],
                         joints['right_e0'], joints['right_e1'],
                         joints['right_w0'], joints['right_w1'],
                         joints['right_w2']])
    else:
        return np.array([joints['left_s0'], joints['left_s1'],
                         joints['left_e0'], joints['left_e1'],
                         joints['left_w0'], joints['left_w1'],
                         joints['left_w2']])
    
def convert_joints_to_dict(joints, limb):
    """Converts a joints numpy array to a joints dict as used by the baxter sdk.
    """
    if limb == 'right':
        return {'right_s0': joints[0], 'right_s1': joints[1], 
                'right_w0': joints[4], 'right_w1': joints[5], 
                'right_w2': joints[6], 'right_e0': joints[2], 
                'right_e1': joints[3]}
    else:
        return {'left_s0': joints[0], 'left_s1': joints[1], 
                'left_w0': joints[4], 'left_w1': joints[5], 
                'left_w2': joints[6], 'left_e0': joints[2], 
                'left_e1': joints[3]}


def get_joint_bounds():
    """Returns array of bounds for each joint.
    These bounds are considered in the optimization algorithm.
    """
    lb = np.radians(np.array([-97.494, -123, -174.987, -2.864, -175.25, -90, -175.25]))
    ub = np.radians(np.array([97.494, 60, 174.987, 150, 175.25, 120, 175.25]))
    return np.array([ (lb[0],ub[0]), (lb[1],ub[1]), (lb[2],ub[2]), 
                      (lb[3],ub[3]), (lb[4],ub[4]), (lb[5],ub[5]), 
                      (lb[6],ub[6]) ])


def angle_between(a, b):
    """Calculates the angle between two vectors.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))


def forward_matrix(joints, goal_frame='right_wrist'):
    """Returns the forward kinematics matrix at the given joints from the "torso" frame to the given goal frame.
    Valid values for the goal frame are:
        * right_arm_mount
        * right_upper_shoulder
        * right_lower_shoulder
        * right_upper_elbow
        * right_lower_elbow
        * right_upper_forearm
        * right_lower_forearm
        * right_wrist
        * right_hand
        * right_gripper_base
        * right_gripper (this frame might slightly differ from yours, check the last transformation to this frame)
        * ... or any of the above with 'right' replaced by 'left'
    Note that the function doesn't check this value for correctness.
        
    If you need other frames, it shouldn't be too difficult to extend this function.
    Just look for the joints in the urdf, add their origin using compose_matrix,
    and the joint itsself using rotation/translation_matrix (for revolute/prismatic joints respectively).
    You can take a look at the s1 joint for an example, and compare values with the corresponding joint in the urdf.
    """
    zaxis = np.array([0,0,1])
    M = np.identity(4)
    
    if goal_frame.startswith('right_'):
        #torso to right_arm_mount
        c = compose_matrix(angles=[0, 0, -0.7854], translate=[0.024645, -0.219645, 0.118588])
        M = np.matmul(M, c)
        if goal_frame == 'right_arm_mount': return M
    else:
        #torso to left_arm_mount
        c = compose_matrix(angles=[0, 0, 0.7854], translate=[0.024645, 0.219645, 0.118588])
        M = np.matmul(M, c)
        if goal_frame == 'left_arm_mount': return M
        
    # Luckily, only the first frame is different between the left and right arm!
    # All following transformations are designed to be the same for both sides.
    
    #{side}_arm_mount to {side}_upper_shoulder, s0 joint
    c = compose_matrix(angles=[0, 0, 0], translate=[0.055695, 0, 0.011038])
    r = rotation_matrix(joints[0], zaxis)
    M = np.matmul(np.matmul(M, c), r)
    if goal_frame.endswith('_upper_shoulder'): return M
    
    #{side}_upper_shoulder to {side}_lower_shoulder, s1 joint
    c = compose_matrix(angles=[-1.57079632679, 0, 0], translate=[0.069, 0, 0.27035])
    r = rotation_matrix(joints[1], zaxis)
    M = np.matmul(np.matmul(M, c), r)
    if goal_frame.endswith('_lower_shoulder'): return M
    
    #{side}_lower_shoulder to {side}_upper_elbow, e0 joint
    c = compose_matrix(angles=[1.57079632679, 0, 1.57079632679], translate=[0.102, 0, 0])
    r = rotation_matrix(joints[2], zaxis)
    M = np.matmul(np.matmul(M, c), r)
    if goal_frame.endswith('_upper_elbow'): return M
    
    #{side}_upper_elbow to {side}_lower_elbow, e1 joint
    c = compose_matrix(angles=[-1.57079632679, -1.57079632679, 0], translate=[0.069, 0, 0.26242])
    r = rotation_matrix(joints[3], zaxis)
    M = np.matmul(np.matmul(M, c), r)
    if goal_frame.endswith('_lower_elbow'): return M
    
    #{side}_lower_elbow to {side}_upper_forearm, w0 joint
    c = compose_matrix(angles=[1.57079632679, 0, 1.57079632679], translate=[0.10359, 0, 0])
    r = rotation_matrix(joints[4], zaxis)
    M = np.matmul(np.matmul(M, c), r)
    if goal_frame.endswith('_upper_forearm'): return M
    
    #{side}_upper_forearm to {side}_lower_forearm, w1 joint
    c = compose_matrix(angles=[-1.57079632679, -1.57079632679, 0], translate=[0.01, 0, 0.2707])
    r = rotation_matrix(joints[5], zaxis)
    M = np.matmul(np.matmul(M, c), r)
    if goal_frame.endswith('_lower_forearm'): return M
    
    #{side}_lower_forearm to {side}_wrist, w2 joint
    c = compose_matrix(angles=[1.57079632679, 0, 1.57079632679], translate=[0.115975, 0, 0])
    r = rotation_matrix(joints[6], zaxis)
    M = np.matmul(np.matmul(M, c), r)
    if goal_frame.endswith('_wrist'): return M
    
    #{side}_wrist to {side}_hand
    c = compose_matrix(angles=[0, 0, 0], translate=[0, 0, 0.11355])
    M = np.matmul(np.matmul(M, c), r)
    if goal_frame.endswith('_hand'): return M
    
    #{side}_hand to {side}_gripper_base
    c = compose_matrix(angles=[0, 0, 0], translate=[0, 0, 0.025])
    M = np.matmul(np.matmul(M, c), r)
    if goal_frame.endswith('_gripper_base'): return M
    
    #{side}_gripper_base to {side}_gripper
    c = compose_matrix(angles=[0, 0, 0], translate=[0, 0, 0.1327])
    M = np.matmul(np.matmul(M, c), r)
    if goal_frame.endswith('_gripper'): return M
    
    return M

def create_pointing_loss(goal_position, limb):
    """Creates a loss that results in the goal pose pointing at a given position.
    The result from IK should give a pose, where the robot is pointing at the goal position using the given limb's gripper,
    event if this position is out of reach.
    
    This serves an example of how to use this file, and also happens to be the main reason it was created.
    Refer to this if you want to make your own custom loss function.
    """
    def pointing_loss(joints):
        M = forward_matrix(joints, limb+'_gripper')
        position = np.array([M[0,3], M[1,3], M[2,3]])
        zaxis = np.array([0,0,1,1])
        #direction that the gripper is pointing at
        pointing_direction = np.matmul(M, zaxis)
        goal_direction = goal_position-position
        angle_deviation = angle_between(pointing_direction[:3], goal_direction)
        #gripper should hold some distance from the goal pose, so that it doesn't touch the goal
        distance_deviation = np.abs( np.linalg.norm(goal_direction) - 0.17 )
        return angle_deviation*angle_deviation+distance_deviation
    return pointing_loss

def create_position_loss(goal_position, limb):
    """Creates a loss that tries to get the gripper as close as possible to the goal position.
    """
    def position_loss(joints):
        M = forward_matrix(joints, limb+'_gripper')
        position = np.array([M[0,3], M[1,3], M[2,3]])
        goal_direction = goal_position-position
        return np.linalg.norm(goal_direction)
    return position_loss

def baxter_ik(loss_func, x0=np.zeros(7), verbose=False):
    """Uses scipy optimization to minimize a given loss function.
    The minimization starts with the joint values x0. For exmaple, these can be set to the current joint values of the robot,
    but the minimization will mostly work fine when setting them to just 0.
    Set verbose to true to print some minimization info.
    
    Example usage:
        >>> goal_position = np.array([1.294, -0.063, 0.003])
        >>> joints = baxter_ik(create_pointing_loss(goal_position))
    """
    res = minimize(loss_func, x0, bounds=get_joint_bounds())
    if verbose: print(res)
    return res.x
    
