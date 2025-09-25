import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import math
import pandas as pd
import csv
from scipy.interpolate import interp1d
import casadi as cs
import os
import scipy


def LonLatDeviations(ego_yaw, ego_x, ego_y, ref_x,ref_y):
    '''
    This method is based on rotating the deviation vectors by the negative 
    of the yaw angle of the vehicle, which aligns the deviation vectors with 
    the longitudinal and lateral axes of the vehicle.
    '''
    rotcos      = np.cos(-ego_yaw)
    rotsin      = np.sin(-ego_yaw)
    dev_long    = rotcos * (ref_x - ego_x) - rotsin * (ref_y - ego_y)
    dev_lat     = rotsin * (ref_x - ego_x) + rotcos * (ref_y - ego_y)
    return dev_long, dev_lat

def LatLonDeviation(ego_x, ego_y, ref_x,ref_y):
    # calculate the deviation vectors
    deviation_x = ego_x - ref_x
    deviation_y = ego_y - ref_y

    # calculate the longitudinal deviations as the magnitude of the deviation vectors
    longitudinal_deviation = np.sqrt(deviation_x**2 + deviation_y**2)

    # calculate the lateral deviations as the magnitude of the cross product of the deviation vectors and the reference trajectory vectors, divided by the magnitude of the reference trajectory vectors.
    lateral_deviation = np.abs(np.cross(np.column_stack((deviation_x, deviation_y)), np.column_stack((ref_x, ref_y)))) / np.sqrt(ref_x**2 + ref_y**2)
    return longitudinal_deviation, lateral_deviation

def postprocess_yaw(yaw):
    if isinstance(yaw, (list, np.ndarray)):
        yaw = np.fmod(yaw, 2*np.pi)
        yaw[yaw < 0] += 2*np.pi
        return yaw
    else:
        yaw = math.fmod(yaw, 2*math.pi)
        if yaw < 0:
            yaw += 2*math.pi
        return yaw

def quat_to_yaw(x, y, z, w):
    """쿼터니언 -> yaw(rad), 범위 (-pi, pi]"""
    # ZYX 순서에서의 yaw 공식
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_2pi(a):
    a = math.fmod(a, 2.0 * math.pi)
    return a + 2.0 * math.pi if a < 0.0 else a


def angle_diff(a, b):
    """(a - b)을 (-pi, pi]로 래핑"""
    d = a - b
    return (d + math.pi) % (2.0 * math.pi) - math.pi

def PlannerEmulator(ref_traj_set, current_pose, N, Tp, loop_circuit):
    
    # 1. Step: calculate the euclidean distance between the current pose and each point in the reference trajectory
    #euclidean_dist = [np.linalg.norm(np.array([ref_traj_set['pos_x'][i],ref_traj_set['pos_y'][i]]) - np.array([current_pose[0],current_pose[1]])) for i in range(len(ref_traj_set['pos_x']))]
    
    a = np.array([current_pose[0:2]])
    b = np.column_stack([ref_traj_set['pos_x'], ref_traj_set['pos_y']])

    dists = scipy.spatial.distance.cdist(a, b)
    closest_point_index = np.argmin(dists)
    
    # find the index of the point in the reference trajectory with the minimum euclidean distance
    #closest_point_index = euclidean_dist.index(min(euclidean_dist))
        # shift closest_point_index to extract trajectory 5 points behind the vehicle: not a good idea
        # closest_point_index -= 5
        # if closest_point_index < 0:
        #     closest_point_index = 0
        
    # 2. Step: extract trajectory indexes that are in the N*Ts temporal horizon 
    temporal_idx= list()
    T = 0
    temporal_idx.append(closest_point_index)
    while T <= Tp:
        # check if we are at the end of the trajectory:
        curr_idx = temporal_idx[-1]
        if curr_idx + 1 >= len(ref_traj_set['pos_x']):
            if loop_circuit:
                temporal_idx.append(0)
            else:
                print("trajectory extraction failed: END OF TRAJECTORY REACHED")
                # return
        else:    
            temporal_idx.append(curr_idx + 1)
        T += np.linalg.norm(np.array([ref_traj_set['pos_x'][temporal_idx[-1]],ref_traj_set['pos_y'][temporal_idx[-1]]]) - np.array([ref_traj_set['pos_x'][temporal_idx[-2]],ref_traj_set['pos_y'][temporal_idx[-2]]]))/ref_traj_set['ref_v'][temporal_idx[-1]]

    # 3. Step: extract the trajectory corresponding to the indexes 
    extracted_traj = {key: [value[i] for i in temporal_idx] for key, value in ref_traj_set.items()}

    # 4. Step: interpolate/extrapolate values to have N traj points with Ts distance
    # Create an array of linearly interpolated values
    if N != len(extracted_traj['pos_x']):
        final_extracted_traj = {}
        for key in extracted_traj.keys():
            interpolated_values = np.interp(np.linspace(0, len(extracted_traj[key])-1, N), np.arange(len(extracted_traj[key])), extracted_traj[key])
            # original reference yaw is defined in [0, 2pi] and jumps from 0 to 2pi and backwards if the interval is exceeded
            # interp generates values in between, which is not correct --> solution:  
            if key == "ref_yaw":
                if (np.abs(np.diff(extracted_traj[key])) > np.deg2rad(250)).any():
                    x = np.linspace(0, len(extracted_traj[key])-1, N)
                    xp = np.arange(len(extracted_traj[key]))
                    fp = extracted_traj[key]
                    period = 2*np.pi
                    interpolated_values = np.mod(np.interp(x, xp, np.unwrap(fp, period=period)), period)
            final_extracted_traj[key] = interpolated_values
    else: 
        final_extracted_traj = extracted_traj
    
    return closest_point_index, final_extracted_traj
