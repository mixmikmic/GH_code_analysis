import rospy,tf, numpy as np, transformations, json, rospkg, yaml
from os import system
from scipy.optimize import minimize
import transformations
import time

rospy.init_node('calibrator')

rospy.set_param('/optitrack/objects', ['/robot/calibrator'])

tfl = tf.TransformListener(True, rospy.Duration(2))  # tf will have 2 seconds of cache

rospack = rospkg.RosPack()

world_frame_robot = rospy.get_param('/optitrack/world_frame') # The world frame of the robot
world_frame_opt = 'optitrack_frame' # The world frame of optitrack
calib_frame_robot = 'right_gripper' # The frame used for calibration in the robot's frame
calib_frame_opt = '/robot/calibrator' # The frame used for calibration in the optitrack's frame

print "Now run the publisher: roslaunch optitrack_publisher optitrack_publisher.launch ip:=<IP> world:={}".format(world_frame_robot)

def record_calibration_points(continuous = True, duration=60, min_dist=0.01, max_dur=0.05):
    mat_robot = [] # Matrix of all calibration points of calib_frame_robot in frame world_frame_robot (position only)
    mat_opt = [] # Matrix of all calibration points of calib_frame_opt in frame world_frame_opt (position only)
    max_dur = rospy.Duration(max_dur) # seconds
    duration = rospy.Duration(duration)
    
    start = rospy.Time.now()
    last_point = None
    entry = ""
    while continuous and rospy.Time.now()<start+duration or not continuous and entry=="":   
        ref_time = tfl.getLatestCommonTime(world_frame_robot, calib_frame_opt)
        now = rospy.Time.now()
        
        if ref_time > now - max_dur:
            try:
                pose_rg_robot = tfl.lookupTransform(world_frame_robot, calib_frame_robot, rospy.Time(0))
            except Exception, e:
                print "Robot <-> Optitrack transformation not available at the last known common time:", e.message
            else:
                if last_point is None or transformations.distance(pose_rg_robot, last_point)>min_dist:
                    try:
                        pose_rg_opt = tfl.lookupTransform(world_frame_opt, calib_frame_opt, rospy.Time(0))
                    except:
                        print "Optitrack Marker not visible at the last know common time"
                    else:
                        mat_robot.append(np.array(pose_rg_robot))
                        mat_opt.append(np.array(pose_rg_opt))
                        last_point = pose_rg_robot
                        system('beep')
        else:
            print "TFs are", (now - ref_time).to_sec(), "sec late"
        
        if continuous:
            rospy.sleep(0.25)
        else:
            entry = raw_input("Press enter to record a new point or q-enter to quit ({} points)".format(len(mat_robot)))
    return mat_opt, mat_robot

def extract_transforms(flat_transforms):
    # a transform is 3 pos and 4 rot
    nb_transform = len(flat_transforms) / 7
    list_transforms = []
    for i in range(nb_transform):
        pose = []
        # extract the pose
        pose.append(flat_transforms[i * 7:i * 7 + 3])
        pose.append(flat_transforms[i * 7 + 3:i * 7 + 7])
        # append it to the list of transforms
        list_transforms.append(pose)
    return list_transforms
        
def result_to_calibration_matrix(result):
    calibration_matrix = transformations.inverse_transform(result)
    return [map(float, calibration_matrix[0]), map(float, calibration_matrix[1].tolist())]

def evaluate_calibration(calibrations, coords_robot, coords_opt):
    def quaternion_cost(norm_coeff):
        C = 0
        for transform in list_calibr:
            # norm of a quaternion is always 1
            C += norm_coeff * abs(np.linalg.norm(transform[1]) - 1)
        return C
        
    def distance_cost(pose1, pose2, rot_coeff=2):
        pos_cost = 0
        # calculate position ditance
        pos_cost = np.linalg.norm(np.array(pose1[0]) - np.array(pose2[0]))
        # distance between two quaternions
        rot_cost = 1 - np.inner(pose1[1], pose2[1])**2
        return pos_cost + rot_coeff * rot_cost

    # first extract the transformations
    list_calibr = extract_transforms(calibrations)
    # set the base transform
    A = list_calibr[0]
    B = list_calibr[1]
    # loop trough all the transforms
    cost = quaternion_cost(1)
    nb_points = len(coords_robot)
    for i in range(nb_points):
        robot = coords_robot[i]
        opt = coords_opt[i]
        product = transformations.multiply_transform(robot, B)
        product = transformations.multiply_transform(A, product)
        product[1] /= np.linalg.norm(product[1])
        cost += distance_cost(opt, product)
    return cost

bounds = []
pos_bounds = [-4, 4]
rot_bounds = [-1, 1]
for i in range(2):
    for j in range(3):
        bounds.append(pos_bounds)
    for j in range(4):
        bounds.append(rot_bounds)

def calculate_position_error(A, B, coords_robot, coords_opt):
    norm = 0.
    # precision error
    for i in range(len(coords_robot)):
        robot = coords_robot[i]
        opt = coords_opt[i]
        product = tranformations.multiply_tranform(robot, B)
        product = tranformations.multiply_tranform(A, product)
        norm += np.linalg.norm(opt[0], product[0])
    return norm

# Record during 60 sec... set continuous=False for an interactive mode
mat_opt, mat_robot = record_calibration_points(continuous=True)

print len(mat_opt), "points recorded"

initial_guess = [0,0,0,0,0,0,1]*2

t0 = time.time()
# Be patient, this cell can be long to execute...
result = minimize(evaluate_calibration, initial_guess, args=(mat_robot, mat_opt, ),
                  method='L-BFGS-B', bounds=bounds)
print time.time()-t0, "seconds of optimization"

print result

result_list = extract_transforms(result.x)

calibration_matrix_a = result_to_calibration_matrix(result_list[0])

rospy.set_param("/optitrack/calibration_matrix", calibration_matrix_a)

with open(rospack.get_path("optitrack_publisher")+"/config/calibration_matrix.yaml", 'w') as f:
    yaml.dump(calibration_matrix_a, f)

calibration_matrix_b = result_to_calibration_matrix(result_list[1])

with open(rospack.get_path("optitrack_publisher")+"/config/calibration_matrix_b.yaml", 'w') as f:
    yaml.dump(calibration_matrix_b, f)

mat_opt_check, mat_robot_check = record_calibration_points(False)

print len(mat_opt), "points recorded"

calculate_cost(result_a, result_b, mat_robot_check, mat_opt_check)

avg_error = calculate_position_error(result_a, result_b, mat_robot_check, mat_opt_check)/len(mat_opt_check)

avg_error

