#!/usr/bin/env python
#
# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
This module contains a local planner to perform
low-level waypoint following based on PID controllers.
"""

from collections import deque
import rospy
import math
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import QuaternionStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from tf.transformations import euler_from_quaternion
from carla_waypoint_types.srv import GetWaypoint
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus
from vehicle_pid_controller import VehiclePIDController  # pylint: disable=relative-import
from misc import distance_vehicle  # pylint: disable=relative-import
import carla
import carla_ros_bridge.transforms as trans
import copy
import cubic_spline_planner

# parameter
MAX_SPEED = 40.0 / 3.6 # maximum speed [m/s]
MAX_ACCEL = 20.0 # maximum acceleration [m/ss]
MAX_CURVATURE = 10.0  # maximum curvature [1/m]
MIN_T = 1.5 # min prediction time [s]
MAX_T = 2.5 # max prediction time [s]
DT = 0.2 # time tick [s]
D_T_S = 5.0 / 3.6 # target speed sampling length [m/s]
N_S_SAMPLE = 1 # sampling number of target speed

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 0.165
K_LON = 1.0
K_GLOBAL = 0.1
K_OBSTACLE = 1000.0

CRASH_DISTANCE = 10.0

GOAL_POINT_X = -36.857
GOAL_POINT_Y = -178.11
GOAL_POINT_Z = 0.0


class QuinticPolynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt

class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt

class CandinateFrenetPath:
    def __init__(self):
        self.t = [] # time
        self.d = [] # lateral position
        self.d_d = [] # lateral velocity
        self.d_dd = [] # lateral acceleration
        self.d_ddd = [] # lateral jerk
        self.s = [] # longituinal position
        self.s_d = [] # longitudinal velocity
        self.s_dd = [] # longitudinal acceleration
        self.s_ddd = [] # longitudinal jerk
        self.lateral_cost = 0.0 # lateral cost
        self.longitudinal_cost = 0.0 # longitudinal cost
        self.global_following_cost = 0.0 # global trajectory following cost
        self.obstacle_distance_cost = 0.0 # obstacle distance cost
        self.final_cost = 0.0 # final cost
        self.path_indicator = 0 # 0: candidate path, 1: selected path, 2: unavailable path

        self.x = [] # x coodrdinate
        self.y = [] # y coordinate
        self.yaw = [] # yaw value
        self.ds = [] # direct distance
        self.curv = [] # curvature

class Obstacle:
    def __init__(self):
        self.id = -1 # actor id
        self.vx = 0.0 # velocity in x direction
        self.vy = 0.0 # velocity in y direction
        self.vz = 0.0 # velocity in z direction
        self.ros_transform = None # transform of the obstacle in ROS coordinate
        self.carla_transform = None # transform of the obstacle in Carla world coordinate
        self.bbox = None # Bounding box w.r.t ego vehicle's local frame
        self.dynamic = 0 # 0: static obstacle, 1: dynamic obstacle

class MyLocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is
    generated on-the-fly. The low-level motion of the vehicle is computed by using two PID
    controllers, one is used for the lateral control and the other for the longitudinal
    control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, role_name, opt_dict=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param role_name: name of the actor
        :param opt_dict: dictionary of arguments with the following semantics:

            target_speed -- desired cruise speed in Km/h

            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I'}

            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal
                                         PID controller
                                         {'K_P':, 'K_D':, 'K_I'}
        """
        self.selected_path = None
        self.target_route_point = None
        self._actual_goal = None
        self._current_waypoint = None
        self._vehicle_controller = None
        self._waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 30
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._vehicle_yaw = None
        self._current_speed = None
        self._current_pose = None
        self._current_accel = None
        self._obstacles = []

        # global path
        self._total_waypoints = [] # total waypoints
        self._total_waypoints_s = 0 # total waypoints s coordinate
        self._wx = [] # waypoints x coordinate
        self._wy = [] # waypoints y coordinate
        # self._csp = None # 2d cubicspline
        # self._tx = [] # x coordinate
        # self._ty = [] # y coordinate
        # self._tyaw = [] # yaw value
        # self._tc = [] # curvature value

        # state
        # self._path_iteration = 0
        self._s_location = 0.0
        self._s_vel = 0.0 # m/s
        self._s_accel = 0.0 # m/ss
        self._d_location = 0.0
        self._d_vel = 0.0 # m/s
        self._d_accel = 0.0 # m/ss

        self._following_frenet_path = 0 # 0: new path plan needed, 1: foloowing selected path
        self._frenet_buffer = deque(maxlen=self._buffer_size)
        self.frenet_route_point = None


        # get world and map for finding actors and waypoints
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        self.world = client.get_world()
        self.map = self.world.get_map()        

        self._target_point_publisher = rospy.Publisher(
            "/next_target", PointStamped, queue_size=1)

        self._candidate_path_publisher = rospy.Publisher(
            "/candidate_path", MarkerArray, queue_size=20)

        self._selected_path_publisher = rospy.Publisher(
            "/selected_path", Marker, queue_size=1)

        self._odometry_subscriber = rospy.Subscriber(
            "/carla/{}/odometry".format(role_name), Odometry, self.odometry_updated)

        self._actual_goal_subscriber = rospy.Subscriber(
            "carla/{}/goal".format(role_name), PoseStamped, self.actual_goal_updated)

        self._vehicle_status_subscriber = rospy.Subscriber(
            "/carla/{}/vehicle_status".format(role_name), CarlaEgoVehicleStatus, self.vehicle_status_updated)

        rospy.wait_for_service('/carla_waypoint_publisher/{}/get_waypoint'.format(role_name))
        self._get_waypoint_client = rospy.ServiceProxy(
            '/carla_waypoint_publisher/{}/get_waypoint'.format(role_name), GetWaypoint)

        # initializing controller
        self._init_controller(opt_dict)

    def odometry_updated(self, odo):
        """
        Callback on new odometry
        """
        self._current_speed = math.sqrt(odo.twist.twist.linear.x ** 2 +
                                        odo.twist.twist.linear.y ** 2 +
                                        odo.twist.twist.linear.z ** 2) * 3.6

        self._current_pose = odo.pose.pose
        quaternion = (
            odo.pose.pose.orientation.x,
            odo.pose.pose.orientation.y,
            odo.pose.pose.orientation.z,
            odo.pose.pose.orientation.w
        )
        _, _, self._vehicle_yaw = euler_from_quaternion(quaternion)

    def vehicle_status_updated(self, vehicle_status):

        self._current_accel = vehicle_status.acceleration

    def actual_goal_updated(self, actual_goal):

        self._actual_goal = actual_goal

    def get_waypoint(self, location):
        """
        Helper to get waypoint from a ros service
        """
        try:
            response = self._get_waypoint_client(location)
            return response.waypoint
        except (rospy.ServiceException, rospy.ROSInterruptException) as e:
            if not rospy.is_shutdown:
                rospy.logwarn("Service call failed: {}".format(e))

    def closest_waypoint_index(self, location, total_waypoints):
        
        min_distance = float("inf")

        for i, waypoint in enumerate(total_waypoints):
            distance  = distance_vehicle(waypoint, location)

            if distance < min_distance:
                closest_waypoint_index = i
                min_distance = distance

        return closest_waypoint_index

    def next_waypoint_index(self, location, total_waypoints):

        closest_waypoint_index = self.closest_waypoint_index(location, total_waypoints)

        if (closest_waypoint_index + 1 == len(total_waypoints)):
            waypoints_vector = [total_waypoints[closest_waypoint_index-1].position.x - total_waypoints[closest_waypoint_index].position.x, 
            total_waypoints[closest_waypoint_index-1].position.y - total_waypoints[closest_waypoint_index].position.y]

            ego_waypoint_vector = [location.x - total_waypoints[closest_waypoint_index].position.x, location.y - total_waypoints[closest_waypoint_index].position.y]

            if np.sign(np.dot(waypoints_vector, ego_waypoint_vector)) >= 0:
                next_waypoint_index = closest_waypoint_index

            else:
                return None

        else:
            waypoints_vector = [total_waypoints[closest_waypoint_index+1].position.x - total_waypoints[closest_waypoint_index].position.x, 
                total_waypoints[closest_waypoint_index+1].position.y - total_waypoints[closest_waypoint_index].position.y]

            ego_waypoint_vector = [location.x - total_waypoints[closest_waypoint_index].position.x, location.y - total_waypoints[closest_waypoint_index].position.y]

            if np.sign(np.dot(waypoints_vector, ego_waypoint_vector)) >= 0:
                next_waypoint_index = closest_waypoint_index+1

            else:
                next_waypoint_index = closest_waypoint_index

        return next_waypoint_index

    def frenet_transform(self, location, total_waypoints):

        next_waypoint_index = self.next_waypoint_index(location, total_waypoints)

        if not next_waypoint_index:
            return 0.0, 0.0, [0.0, 0.0, 0.0]

        prev_waypoint_index = next_waypoint_index - 1

        waypoints_vector = [total_waypoints[next_waypoint_index].position.x - total_waypoints[prev_waypoint_index].position.x, 
            total_waypoints[next_waypoint_index].position.y - total_waypoints[prev_waypoint_index].position.y, 0.0]

        ego_waypoint_vector = [location.x - total_waypoints[prev_waypoint_index].position.x, location.y - total_waypoints[prev_waypoint_index].position.y, 0.0]

        if (waypoints_vector[0] == 0) and (waypoints_vector[1] == 0):
            waypoints_vector = [total_waypoints[next_waypoint_index].position.x - total_waypoints[next_waypoint_index-2].position.x, 
            total_waypoints[next_waypoint_index].position.y - total_waypoints[next_waypoint_index-2].position.y, 0.0]

        proj_norm = (ego_waypoint_vector[0]*waypoints_vector[0] + ego_waypoint_vector[1]*waypoints_vector[1]) / (waypoints_vector[0]**2 + waypoints_vector[1]**2)
        proj_x = waypoints_vector[0] * proj_norm
        proj_y = waypoints_vector[1] * proj_norm

        d_coordinate = np.sqrt((ego_waypoint_vector[0] - proj_x) ** 2 +  (ego_waypoint_vector[1] - proj_y) ** 2)
        d_cross_product = np.cross(ego_waypoint_vector, waypoints_vector)

        if d_cross_product[-1] > 0:
            d_coordinate = -d_coordinate
        else:
            d_coordinate = d_coordinate

        s_coordinate = 0.0
        for i in range(prev_waypoint_index):
            s_coordinate = s_coordinate + np.sqrt((total_waypoints[i+1].position.x - total_waypoints[i].position.x) ** 2 + (total_waypoints[i+1].position.y - total_waypoints[i].position.y) ** 2)
            

        s_coordinate = s_coordinate + np.sqrt(proj_x ** 2 + proj_y ** 2)

        return s_coordinate, d_coordinate, waypoints_vector

    def cartesian_transform(self, s_coordinate, d_coordinate, total_waypoints, total_waypoints_s):

        previous_waypoint_index = 0

        while (s_coordinate > total_waypoints_s[previous_waypoint_index + 1]) and (previous_waypoint_index < len(total_waypoints_s) - 2):
            previous_waypoint_index = previous_waypoint_index + 1

        next_waypoint_index = previous_waypoint_index + 1

        dx = (total_waypoints[next_waypoint_index].position.x - total_waypoints[previous_waypoint_index].position.x)
        dy = (total_waypoints[next_waypoint_index].position.y - total_waypoints[previous_waypoint_index].position.y)

        s_direction = np.arctan2(dy, dx)

        s_segment = s_coordinate - total_waypoints_s[previous_waypoint_index]

        x_segment = total_waypoints[previous_waypoint_index].position.x + s_segment * np.cos(s_direction)
        y_segment = total_waypoints[previous_waypoint_index].position.y + s_segment * np.sin(s_direction)

        s_perpendicular = s_direction + 90 * np.pi/180

        x_coordinate = x_segment + d_coordinate * np.cos(s_perpendicular)
        y_coordinate = y_segment + d_coordinate * np.sin(s_perpendicular)

        return x_coordinate, y_coordinate, s_direction

    def calc_frenet_paths(self, s_location, s_vel, s_accel, d_location, d_vel, d_accel, location, target_speed):
        frenet_paths = []

        current_waypoint = self.get_waypoint(location)
        road_width = current_waypoint.lane_width

        # generate path to each offset goal

        for di in np.arange(-road_width, road_width+1, road_width):

            # Lateral motion planning
            for Ti in np.arange(MIN_T, MAX_T, DT):
                
                fp = CandinateFrenetPath()

                # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
                lat_qp = QuinticPolynomial(d_location, d_vel, d_accel, di, 0.0, 0.0, Ti)

                fp.t = [t for t in np.arange(0.0, Ti, DT)]
                fp.d = [lat_qp.calc_point(t) for t in fp.t]
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                # Longitudinal motion planning (Velocity keeping)
                for tv in np.arange((target_speed * 1 / 3.6) - D_T_S * N_S_SAMPLE, (target_speed * 1 / 3.6) + D_T_S * N_S_SAMPLE, D_T_S):
                    tfp = copy.deepcopy(fp)
                    lon_qp = QuarticPolynomial(s_location, s_vel, s_accel, tv, 0.0, Ti)

                    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                    Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                    Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                    # square of diff from target speed
                    ds = ((target_speed * 1 / 3.6) - tfp.s_d[-1]) ** 2

                    tfp.lateral_cost = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                    tfp.longitudinal_cost = K_J * Js + K_T * Ti + K_D * ds

                    # print("\x1b[6;30;33m------target_s------\x1b[0m")
                    # print(tfp.s)

                    # print("\x1b[6;30;33m------target_d------\x1b[0m")
                    # print(tfp.d)

                    frenet_paths.append(tfp)

        # print("calc_frenet_paths number", len(frenet_paths))

        return frenet_paths

    def calc_global_paths(self, fplist, total_waypoints, total_waypoints_s):
        for fp in fplist:

            # print("fp_s_length: ", len(fp.s))

            # calc global positions
            for i in range(len(fp.s)):
                x_coordinate, y_coordinate, _ = self.cartesian_transform(fp.s[i], fp.d[i], total_waypoints, total_waypoints_s)

                fp.x.append(x_coordinate)
                fp.y.append(y_coordinate)


                # # print("fp_s_value: ", fp.s[i])
                # ix, iy = csp.calc_position(fp.s[i])

                # # print("ix: ", ix)
                # # print("iy: ", iy)
                # if ix is None:
                #     break
                # i_yaw = csp.calc_yaw(fp.s[i])
                # di = fp.d[i]
                # fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
                # fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
                # fp.x.append(fx)
                # fp.y.append(fy)

            # print("fp_x_length: ", len(fp.x))

            if (len(fp.x) <= 1):
                continue
                
            # calc yaw and ds
            for i in range(len(fp.x) - 1):
                dx = fp.x[i + 1] - fp.x[i]
                dy = fp.y[i + 1] - fp.y[i]
                fp.yaw.append(math.atan2(dy, dx))
                fp.ds.append(math.hypot(dx, dy))

            # print("fp_yaw: ", fp.yaw)
            fp.yaw.append(fp.yaw[-1])
            fp.ds.append(fp.ds[-1])

            # calc curvature
            for i in range(len(fp.yaw) - 1):
                fp.curv.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

            # print("fp_curv: ", fp.curv)

        # print("calc_global_paths number", len(fplist))

        return fplist

    def calc_global_following(self, fplist):
        path_point = Point()
        
        for i, _ in enumerate(fplist):

            path_offset = 0.0

            for d_value in fplist[i].d:
                path_offset = path_offset + abs(d_value)

            fplist[i].global_following_cost = path_offset

        # print("calc_global_following number", len(fplist))
        return fplist

    def get_obstacles(self, location, range):
        """
        Get a list of obstacles that are located within a certain distance from the location.
        
        :param      location: queried location
        :param      range: search distance from the queried location
        :type       location: geometry_msgs/Point
        :type       range: float or double
        :return:    None
        :rtype:     None
        """
        self._obstacles = []
        actor_list = self.world.get_actors()
        for actor in actor_list:
            if "role_name" in actor.attributes:
                if actor.attributes["role_name"] == 'autopilot' or actor.attributes["role_name"] == "static":
                    carla_transform = actor.get_transform()
                    ros_transform = trans.carla_transform_to_ros_pose(carla_transform)
                    x = ros_transform.position.x
                    y = ros_transform.position.y
                    z = ros_transform.position.z 
                    distance = math.sqrt((x-location.x)**2 + (y-location.y)**2)
                    if distance < range:
                        # print("obs distance: {}").format(distance)
                        ob = Obstacle()
                        ob.id = actor.id
                        ob.carla_transform = carla_transform
                        ob.ros_transform = ros_transform
                        ob.vx = actor.get_velocity().x
                        ob.vy = actor.get_velocity().y
                        ob.vz = actor.get_velocity().z
                        ob.bbox = actor.bounding_box # in local frame
                        if (ob.vx > 0 or ob.vy > 0 or ob.vz > 0):
                            ob.dynamic = 1
                        else:
                            ob.dynamic = 0
                        # print("x: {}, y: {}, z:{}").format(x, y, z)
                        # print("bbox x:{} y:{} z:{} ext: {} {} {}".format(ob.bbox.location.x, ob.bbox.location.y, ob.bbox.location.z, ob.bbox.extent.x, ob.bbox.extent.y, ob.bbox.extent.z))
                        self._obstacles.append(ob)

    def check_obstacle(self, point, polygon):
        """
        Check whether a point is inside the bounding box of the obstacle

        :param      point: a location to check the collision (in ROS frame)
        :param      obstacle: an obstacle for collision check
        :type       point: geometry_msgs/Point
        :type       obstacle: object Obstacle
        :return:    true or false
        :rtype:     boolean   
        """
        carla_location = carla.Location()
        carla_location.x = point.x
        carla_location.y = -point.y
        carla_location.z = point.z

        N = len(polygon) - 1
        counter = 0
        p1 = polygon[0]

        for i in range(1, N+1):
            p2 = polygon[i%N]

            if carla_location.y > min(p1[1], p2[1]) and carla_location.y <= max(p1[1], p2[1]) and carla_location.x <= max(p1[0], p2[0]) and p1[1] != p2[1]:
                xinters = (carla_location.y-p1[1])*(p2[0]-p1[0])/(p2[1]-p1[1]) + p1[0]
                
                if(p1[0]==p2[0] or carla_location.x<=xinters):
                    counter += 1

            p1 = p2

        if counter % 2 == 0:
            res = 0

        else:
            res = 1

        return res


        # vx = [v.x for v in vertices]
        # vy = [v.y for v in vertices]
        # vz = [v.z for v in vertices]
        # return carla_location.x >= min(vx) and carla_location.x <= max(vx) \
        #         and carla_location.y >= min(vy) and carla_location.y <= max(vy) \
        #         and carla_location.z >= min(vz) and carla_location.z <= max(vz) 

    def check_collision(self, fp, location):

        actor_list = self.world.get_actors()
        for actor in actor_list:
            if "role_name" in actor.attributes:
                if actor.attributes["role_name"] == 'ego_vehicle':
                    ego_vehicle = actor
                    # carla_transform = ego_vehicle.get_transform()

                    ego_vehicle_extent_x = ego_vehicle.bounding_box.extent.x
                    ego_vehicle_extent_y = ego_vehicle.bounding_box.extent.y

        min_dist = []
        dist_total = 0.0

        # print("\x1b[6;30;33m------obstacle_number------\x1b[0m")  
        # print(len(self._obstacles))

        if len(self._obstacles) == 0:
            fp.obstacle_distance_cost = 0.0
            return True

        else:
            for obs in self._obstacles:

                if not obs.dynamic:

                    obs_bbox = carla.BoundingBox()
                    obs_bbox.location.x = copy.deepcopy(obs.bbox.location.x)
                    obs_bbox.location.y = copy.deepcopy(obs.bbox.location.y)
                    obs_bbox.location.z = copy.deepcopy(obs.bbox.location.z)
                    obs_bbox.extent.x = copy.deepcopy(obs.bbox.extent.x)
                    obs_bbox.extent.y = copy.deepcopy(obs.bbox.extent.y)
                    obs_bbox.extent.z = copy.deepcopy(obs.bbox.extent.z)
                    obs_bbox.rotation.pitch = copy.deepcopy(obs.bbox.rotation.pitch)
                    obs_bbox.rotation.yaw = copy.deepcopy(obs.bbox.rotation.yaw)
                    obs_bbox.rotation.roll = copy.deepcopy(obs.bbox.rotation.roll)
                    carla_transform = obs.carla_transform

                    obs_bbox.extent.x += ego_vehicle_extent_x+0.1
                    obs_bbox.extent.y += ego_vehicle_extent_y+0.2

                    vertices = obs_bbox.get_world_vertices(carla_transform)

                    polygon = []
                    for v in vertices:
                        if (v.z <= 0.5):
                            polygon.append([v.x, v.y])

                    dist_list = []

                    for (ix, iy) in zip(fp.x, fp.y):
                    
                        path_point = Point()
                        path_point.x = ix
                        path_point.y = iy
                        path_point.z = 0.0

                        dist_list.append(np.sqrt((ix - carla_transform.location.x) ** 2 + (-iy - carla_transform.location.y) ** 2))
                        # dist_total +=  (ix - carla_transform.location.x) ** 2 + (-iy - carla_transform.location.y) ** 2

                        if self.check_obstacle(path_point, polygon):
                            return False

                    min_dist.append(min(dist_list))

                else:

                    t = np.arange(0.0, 0.9, 0.2)

                    obs_bbox = carla.BoundingBox()
                    obs_bbox.location.x = copy.deepcopy(obs.bbox.location.x)
                    obs_bbox.location.y = copy.deepcopy(obs.bbox.location.y)
                    obs_bbox.location.z = copy.deepcopy(obs.bbox.location.z)
                    obs_bbox.extent.x = copy.deepcopy(obs.bbox.extent.x)
                    obs_bbox.extent.y = copy.deepcopy(obs.bbox.extent.y)
                    obs_bbox.extent.z = copy.deepcopy(obs.bbox.extent.z)
                    obs_bbox.rotation.pitch = copy.deepcopy(obs.bbox.rotation.pitch)
                    obs_bbox.rotation.yaw = copy.deepcopy(obs.bbox.rotation.yaw)
                    obs_bbox.rotation.roll = copy.deepcopy(obs.bbox.rotation.roll)
                    carla_transform = obs.carla_transform

                    obs_bbox.extent.x += ego_vehicle_extent_x+0.1
                    obs_bbox.extent.y += ego_vehicle_extent_y+0.2

                    vertices = obs_bbox.get_world_vertices(carla_transform)

                    dist_list = []

                    for t_value in t:

                        polygon = []
                        for v in vertices:
                            if (v.z <= 0.5):
                                moving_v_x = v.x + obs.vx * t_value
                                moving_v_y = v.y + obs.vy * t_value
                                polygon.append([moving_v_x, moving_v_y])


                        for (ix, iy) in zip(fp.x, fp.y):

                            path_point = Point()
                            path_point.x = ix
                            path_point.y = iy
                            path_point.z = 0.0

                            if self.check_obstacle(path_point, polygon):
                                return False

                            dist_list.append(np.sqrt((ix - carla_transform.location.x) ** 2 + (-iy - carla_transform.location.y) ** 2))

                    min_dist.append(min(dist_list))

            fp.obstacle_distance_cost = min(min_dist)
                    
            return True

    def get_coordinate_lanemarking(self, position):
        """
        Helper to get adjacent waypoint 2D coordinates of the left and right lane markings 
        with respect to the closest waypoint
        
        :param      position: queried position
        :type       position: geometry_msgs/Point
        :return:    left and right waypoint in numpy array
        :rtype:     tuple of geometry_msgs/Point (left), geometry_msgs/Point (right)
        """
        # get waypoints along road
        current_waypoint = self.get_waypoint(position)
        waypoint_xodr = self.map.get_waypoint_xodr(current_waypoint.road_id, current_waypoint.lane_id, current_waypoint.s)
        
        # find two orthonormal vectors to the direction of the lane
        yaw = math.pi - waypoint_xodr.transform.rotation.yaw * math.pi / 180.0
        norm_v = np.array([math.cos(yaw), math.sin(yaw)])
        # v = np.array([1.0, math.tan(yaw)])
        # norm_v = v / np.linalg.norm(v)
        right_v = np.array([-norm_v[1], norm_v[0]])
        left_v = np.array([norm_v[1], -norm_v[0]])
        
        # find two points that are on the left and right lane markings
        half_width = current_waypoint.lane_width / 2.0
        left_waypoint = np.array([current_waypoint.pose.position.x, current_waypoint.pose.position.y]) + half_width * left_v
        right_waypoint = np.array([current_waypoint.pose.position.x, current_waypoint.pose.position.y]) + half_width * right_v
        ros_left_waypoint = Point()
        ros_right_waypoint = Point()
        ros_left_waypoint.x = left_waypoint[0]
        ros_left_waypoint.y = left_waypoint[1]
        ros_right_waypoint.x = right_waypoint[0]
        ros_right_waypoint.y = right_waypoint[1]
        return ros_left_waypoint, ros_right_waypoint, left_v, right_v    

    def check_lanes(self, fp, location):

        actor_list = self.world.get_actors()
        for actor in actor_list:
            if "role_name" in actor.attributes:
                if actor.attributes["role_name"] == 'ego_vehicle':
                    ego_vehicle = actor
                    # carla_transform = ego_vehicle.get_transform()

                    ego_vehicle_extent_y = ego_vehicle.bounding_box.extent.y

        current_waypoint = self.get_waypoint(location)

        lane_width = current_waypoint.lane_width
        lane_change = current_waypoint.lane_change

        # print("\x1b[6;30;33m------lane_change------\x1b[0m")
        # print(lane_change)

        # print("\x1b[6;30;33m------left_lane_marking------\x1b[0m")
        # print(current_waypoint.left_lane_marking.type)
        # print("\x1b[6;30;33m------right_lane_marking------\x1b[0m")
        # print(current_waypoint.right_lane_marking.type)

        left_lane_type = current_waypoint.left_lane_marking.type
        right_lane_type = current_waypoint.right_lane_marking.type

        left_not_allowed = (left_lane_type == "Solid") or (left_lane_type == "SolidSolid") or (left_lane_type == "BrokenSolid")
        right_not_allowed = (right_lane_type == "Solid") or (right_lane_type == "SolidSolid") or (right_lane_type == "SolidBroken")

        # print("\x1b[6;30;33m------not_allowed------\x1b[0m")
        # print("left_not_allowed", left_not_allowed)
        # print("right_not_allowed", right_not_allowed)
        
        left_lane_coordinate, right_lane_coordinate, left_v, right_v = self.get_coordinate_lanemarking(location)

        left_allowed_distance = np.sqrt((left_lane_coordinate.x - location.x) ** 2 + (left_lane_coordinate.y - location.y) ** 2) - ego_vehicle_extent_y
        right_allowed_distance = np.sqrt((right_lane_coordinate.x - location.x) ** 2 + (right_lane_coordinate.y - location.y) ** 2) - ego_vehicle_extent_y
        # print("\x1b[6;30;33m------allowed_distance------\x1b[0m")
        # print("left_allowed_distance", left_allowed_distance)
        # print("right_allowed_distance", right_allowed_distance)

        start_d = fp.d[0]
        offset_d = [path_d - start_d for path_d in fp.d]

        # print("\x1b[6;30;33m------d_offset_allowed------\x1b[0m")
        # print(d_offset_allowed)

        if left_not_allowed or lane_change == "Right":
            if any([offset_d_value > left_allowed_distance for offset_d_value in offset_d]):
                # print("\x1b[6;30;33m------offset_d_value------\x1b[0m")
                # print(offset_d_value)
                return False

        if right_not_allowed or lane_change == "Left":
            if any([offset_d_value < -right_allowed_distance for offset_d_value in offset_d]):
                # print("\x1b[6;30;33m------offset_d_value------\x1b[0m")
                # print(offset_d_value)
                return False

        if (left_not_allowed and right_not_allowed) or (lane_change == "NONE"):
            if any([(offset_d_value < -right_allowed_distance) or (offset_d_value > left_allowed_distance) for offset_d_value in offset_d]):
                # print("\x1b[6;30;33m------offset_d_value------\x1b[0m")
                # print(offset_d_value)
                return False



        return True

    def visualize_paths(self, fplist):

        paths = MarkerArray()

        for i in range(len(fplist)):
            path = Marker()
            path.header.frame_id = "map"
            path.ns = 'candidate path'
            path.id = i
            path.type = 4 # LINE_STRIP
            path.action = 0
            path.scale.x = 0.1

            if (fplist[i].path_indicator == 0):
                path.color.g = 1.0

            else:
                path.color.r = 1.0

            path.color.a = 1.0

            for j in range(len(fplist[i].x)):
                point = Point()
                point.x = fplist[i].x[j]
                point.y = fplist[i].y[j]
                path.points.append(point)

            paths.markers.append(path)

        self._candidate_path_publisher.publish(paths)

    def visualize_selected_path(self, fp):

        path = Marker()

        path.header.frame_id = "map"
        path.ns = 'selected path'
        path.type = 4 # LINE_STRIP
        path.action = 0
        path.scale.x = 0.5

        path.color.b = 1.0
        path.color.a = 1.0

        for i in range(len(fp.x)):
            point = Point()
            point.x = fp.x[i]
            point.y = fp.x[i]
            path.points.append(point)

        self._selected_path_publisher.publish(path)
        
    def check_paths(self, fplist, location):
        ok_ind = []
        speed_violation_num = 0
        accel_violation_num = 0
        curv_violation_num = 0
        lane_violation_num = 0
        collision_violation_num = 0

        for i, _ in enumerate(fplist):
            # print("path spped: ", fplist[i].s_d)
            # print("path accel: ", fplist[i].s_dd)
            # print("path curv: ", fplist[i].curv)

            if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
                fplist[i].path_indicator = 2
                speed_violation_num += 1

                continue
            elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
                fplist[i].path_indicator = 2
                accel_violation_num += 1

                continue
            elif any([abs(c) > MAX_CURVATURE for c in fplist[i].curv]):  # Max curvature check
                fplist[i].path_indicator = 2
                curv_violation_num += 1

                continue

            elif not self.check_lanes(fplist[i], location):
                fplist[i].path_indicator = 2
                lane_violation_num += 1

                continue

            elif not self.check_collision(fplist[i], location):
                fplist[i].path_indicator = 2
                collision_violation_num += 1

                continue

            ok_ind.append(i)

        # print("\x1b[6;30;33m------speed_violation_num------\x1b[0m")
        # print(speed_violation_num)

        # print("\x1b[6;30;33m------accel_violation_num------\x1b[0m")
        # print(accel_violation_num)

        # print("\x1b[6;30;33m------curv_violation_num------\x1b[0m")
        # print(curv_violation_num)

        # print("\x1b[6;30;33m------lane_violation_num------\x1b[0m")
        # print(lane_violation_num)

        # print("\x1b[6;30;33m------collision_violation_num------\x1b[0m")
        # print(collision_violation_num)

        # print("check_paths number", len(ok_ind))

        # return [fplist[i] for i in ok_ind]
        return fplist

    def frenet_optimal_planning(self, s_location, s_vel, s_accel, d_location, d_vel, d_accel, location, target_speed, total_waypoints, total_waypoints_s):
        fplist = self.calc_frenet_paths(s_location, s_vel, s_accel, d_location, d_vel, d_accel, location, target_speed)
        fplist = self.calc_global_paths(fplist, total_waypoints, total_waypoints_s)
        fplist = self.calc_global_following(fplist)
        fplist = self.check_paths(fplist, location)

        self.visualize_paths(fplist)

        # print("fplist_length: ", len(fplist))

        # find minimum cost path
        min_cost = float("inf")
        best_path = None

        for fp in fplist:
            if (fp.path_indicator == 2):
                continue

            if (fp.obstacle_distance_cost == 0):
                fp.final_cost = K_LAT * fp.lateral_cost + K_LON * fp.longitudinal_cost + K_GLOBAL * fp.global_following_cost

            else:
                fp.final_cost = K_LAT * fp.lateral_cost + K_LON * fp.longitudinal_cost + K_OBSTACLE * (1.0/fp.obstacle_distance_cost) + K_GLOBAL * fp.global_following_cost 

            # print("\x1b[6;30;33m------cost------\x1b[0m")
            # print("fp_lateral_cost: ", K_LAT * fp.lateral_cost)
            # print("fp_longitudinal_cost: ", K_LON * fp.longitudinal_cost)
            # print("fp_global_following_cost: ", K_GLOBAL * fp.global_following_cost)
            # if (fp.obstacle_distance_cost != 0):
            #     print("fp_obstacle_distance_cost: ", K_OBSTACLE * (1.0/fp.obstacle_distance_cost))
            # print("fp_cost: ", fp.final_cost)

            if min_cost >= fp.final_cost:
                min_cost = fp.final_cost
                best_path = fp

        if best_path:
            # print("\x1b[6;30;33m------best_path_cost------\x1b[0m")
            # print("best_path_lateral_cost: ", K_LAT * best_path.lateral_cost)
            # print("best_path_longitudinal_cost: ", K_LON * best_path.longitudinal_cost)
            # print("best_path_global_following_cost: ", K_GLOBAL * best_path.global_following_cost)
            # if (best_path.obstacle_distance_cost != 0):
            #     print("best_path_obstacle_distance_cost: ", K_OBSTACLE * (1.0/best_path.obstacle_distance_cost))
            # print("best_path_cost: ", best_path.final_cost)
            best_path.path_indicator = 1

        return best_path

    def generate_target_course(self, x, y):
        csp = cubic_spline_planner.CubicSpline2D(x, y)
        s = np.arange(0, csp.s[-1], 0.1)

        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = csp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(csp.calc_yaw(i_s))
            rk.append(csp.calc_curvature(i_s))

        return rx, ry, ryaw, rk, csp

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4}
        args_longitudinal_dict = {
            'K_P': 0.2,
            'K_D': 0.05,
            'K_I': 0.1}

        # parameters overload
        if opt_dict:
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        self._vehicle_controller = VehiclePIDController(args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict)

    def set_global_plan(self, current_plan):
        """
        set a global plan to follow
        """
        self.target_route_point = None
        self._waypoint_buffer.clear()
        self._waypoints_queue.clear()
        for elem in current_plan:
            self._waypoints_queue.append(elem.pose)

        if self._waypoints_queue:
            self._waypoints_queue.append(self._actual_goal.pose)

            for i in range (len(self._waypoints_queue)):
                self._total_waypoints.append(self._waypoints_queue[i])

                self._wx.append(self._waypoints_queue[i].position.x)
                self._wy.append(self._waypoints_queue[i].position.y)

            # self._tx, self. _ty, self._tyaw, self._tc, self._csp = self.generate_target_course(self._wx, self._wy)

            self._total_waypoints_s = np.zeros(len(self._total_waypoints))
            for i in range (len(self._total_waypoints)):
                #print(i)
                self._total_waypoints_s[i], _, _ = self.frenet_transform(self._total_waypoints[i].position, self._total_waypoints)  

                # print("\x1b[6;30;33m------total_waypoint_s_d------\x1b[0m")
                # print(self._total_waypoints_s[i], d)  

            # self._total_waypoints_s[-1] = self._total_waypoints_s[-2] \
            #     + np.sqrt((self._total_waypoints[-1].position.x - self._total_waypoints[-2].position.x) ** 2 + (self._total_waypoints[-1].position.y - self._total_waypoints[-2].position.y) ** 2)


    def run_step(self, target_speed, current_speed, current_pose):
        """
        Execute one step of local planning which involves running the longitudinal
        and lateral PID controllers to follow the waypoints trajectory.
        """
        if not self._waypoint_buffer and not self._waypoints_queue:

            distance_to_goal = np.sqrt((self._actual_goal.pose.position.x - current_pose.position.x) **2 + (self._actual_goal.pose.position.y - current_pose.position.y) **2)

            if distance_to_goal > 3.0:

                target_speed = 10.0

                # move using PID controllers
                control = self._vehicle_controller.run_step(
                    target_speed, current_speed, current_pose, self._actual_goal.pose)

                return control, False

            else: 
                control = CarlaEgoVehicleControl()
                control.steer = 0.0
                control.throttle = 0.0
                control.brake = 1.0
                control.hand_brake = False
                control.manual_gear_shift = False

                rospy.loginfo("Route finished.")

                return control, True

        #   Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        if not self._frenet_buffer:
            self._following_frenet_path = 0

        # current vehicle waypoint
        self._current_waypoint = self.get_waypoint(current_pose.position)

        # get a list of obstacles surrounding the ego vehicle
        self.get_obstacles(current_pose.position, 50.0)

        # Example 1: get two waypoints on the left and right lane marking w.r.t current pose
        # left, right = self.get_coordinate_lanemarking(current_pose.position)
        # print("\x1b[6;30;33m------Example 1------\x1b[0m")
        # print("Left: {}, {}; right: {}, {}".format(left.x, left.y, right.x, right.y))
        
        # # Example 2: check obstacle collision
        # print("\x1b[6;30;33m------Example 2------\x1b[0m")
        # point = Point()
        # point.x = 100.0
        # point.y = 100.0
        # point.z = 1.5
        # for ob in self._obstacles:
        #     print("id: {}, collision: {}".format(ob.id, self.check_obstacle(point, ob)))

        self.target_route_point = self._waypoint_buffer[0]

        if not self._following_frenet_path:

            self._s_location, self._d_location, waypoints_vector = self.frenet_transform(current_pose.position, self._total_waypoints)

            dx = waypoints_vector[0]
            dy = waypoints_vector[1]

            s_direction = np.arctan2(dy, dx)
            s_perpendicular = s_direction + 90 * np.pi/180

            self._s_vel = current_speed * 1/3.6 * np.cos(self._vehicle_yaw - s_direction)
            self._d_vel = current_speed * 1/3.6 * np.sin(self._vehicle_yaw - s_direction)

            accel_x = self._current_accel.linear.x
            accel_y = self._current_accel.linear.y

            self._s_accel = accel_x * np.cos(s_direction) + accel_y * np.sin(s_direction)
            self._d_accel = accel_x * np.cos(s_perpendicular) + accel_y * np.sin(s_perpendicular)

            self.selected_path = self.frenet_optimal_planning(self._s_location, self._s_vel, self._s_accel, self._d_location, self._d_vel, self._d_accel, current_pose.position, target_speed, self._total_waypoints, self._total_waypoints_s)

            if self.selected_path:
                # print("\x1b[6;30;33m------path_exists!------\x1b[0m")

                self._following_frenet_path = 1
                self.visualize_selected_path(self.selected_path)

                for i in range(1, len(self.selected_path.x)):
                    frenet_point = copy.deepcopy(self.target_route_point)
                    frenet_point.position.x = self.selected_path.x[i]
                    frenet_point.position.y = self.selected_path.y[i]

                    self._frenet_buffer.append(frenet_point)

                self.frenet_route_point = self._frenet_buffer[0]

                target_point = PointStamped()
                target_point.header.frame_id = "map"
                target_point.point.x = self.frenet_route_point.position.x
                target_point.point.y = self.frenet_route_point.position.y
                target_point.point.z = self.frenet_route_point.position.z
                self._target_point_publisher.publish(target_point)

                # move using PID controllers
                control = self._vehicle_controller.run_step(
                    target_speed, current_speed, current_pose, self.frenet_route_point)

            else:
                # print("\x1b[6;30;33m------path_not available------\x1b[0m")
                self._following_frenet_path = 0

                target_point = PointStamped()
                target_point.header.frame_id = "map"
                target_point.point.x = self.target_route_point.position.x
                target_point.point.y = self.target_route_point.position.y
                target_point.point.z = self.target_route_point.position.z
                self._target_point_publisher.publish(target_point)

                target_speed = 10

                # move using PID controllers
                control = self._vehicle_controller.run_step(
                    target_speed, current_speed, current_pose, self.target_route_point)

                for obs in self._obstacles:
                    distance = np.sqrt((current_pose.position.x - obs.ros_transform.position.x) ** 2 + (current_pose.position.y - obs.ros_transform.position.y) ** 2)
                    
                    # print("\x1b[6;30;33m------obstacle_distance------\x1b[0m")
                    # print(distance)


                    if distance < CRASH_DISTANCE:
                        control.brake = 1.0
                        break
        
        else:
            self.visualize_selected_path(self.selected_path)
            self.frenet_route_point = self._frenet_buffer[0]

            selected_path_copy = copy.deepcopy(self.selected_path)

            path_lane_check = self.check_lanes(selected_path_copy, current_pose.position)

            # counter = 0
            # while (self.frenet_route_point.position.x != selected_path_copy.x[counter]):
            #     counter += 1
                
            # for i in range(counter):
            #     selected_path_copy.x.pop(0)

            path_collision_check = self.check_collision(selected_path_copy, current_pose.position)

            if (not path_lane_check) or (not path_collision_check):

                # print("\x1b[6;30;33m------path_not available------\x1b[0m")

                self._frenet_buffer.clear()
                self._following_frenet_path = 0

                target_point = PointStamped()
                target_point.header.frame_id = "map"
                target_point.point.x = self.target_route_point.position.x
                target_point.point.y = self.target_route_point.position.y
                target_point.point.z = self.target_route_point.position.z
                self._target_point_publisher.publish(target_point)

                target_speed = 10

                # move using PID controllers
                control = self._vehicle_controller.run_step(
                    target_speed, current_speed, current_pose, self.target_route_point)

                for obs in self._obstacles:
                    distance = np.sqrt((current_pose.position.x - obs.ros_transform.position.x) ** 2 + (current_pose.position.y - obs.ros_transform.position.y) ** 2)
                    
                    # print("\x1b[6;30;33m------obstacle_distance------\x1b[0m")
                    # print(distance)


                    if distance < CRASH_DISTANCE:
                        control.brake = 1.0
                        break

            else:

                target_point = PointStamped()
                target_point.header.frame_id = "map"
                target_point.point.x = self.frenet_route_point.position.x
                target_point.point.y = self.frenet_route_point.position.y
                target_point.point.z = self.frenet_route_point.position.z
                self._target_point_publisher.publish(target_point)

                # move using PID controllers
                control = self._vehicle_controller.run_step(
                    target_speed, current_speed, current_pose, self.frenet_route_point)


        # purge the queue of obsolete waypoints
        max_index_global = -1
        max_index_frenet = -1

        sampling_radius = target_speed * 1 / 3.6  # 1 seconds horizon
        min_distance = sampling_radius * self.MIN_DISTANCE_PERCENTAGE

        for i, route_point in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    route_point, current_pose.position) < min_distance:
                max_index_global = i
        if max_index_global >= 0:
            for i in range(max_index_global + 1):
                self._waypoint_buffer.popleft()



        for i, route_point in enumerate(self._frenet_buffer):
            if distance_vehicle(
                    route_point, current_pose.position) < min_distance:
                max_index_frenet = i
        if max_index_frenet >= 0:
            for i in range(max_index_frenet + 1):
                self._frenet_buffer.popleft()

        
        return control, False
