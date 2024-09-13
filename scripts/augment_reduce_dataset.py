import numpy as np
import os


def interpolate_between_all_rows(arr, k):
    """
    Linearly interpolates between every pair of consecutive rows in a NumPy array.

    Parameters:
        arr (np.ndarray): The input array of shape (n, 6).
        k (int): Number of steps to interpolate between each pair of consecutive rows.
    
    Returns:
        np.ndarray: Array with interpolated values between each pair of consecutive rows.
    """
    interpolated_result = []

    # Iterate over every consecutive pair of rows
    for i in range(len(arr) - 1):
        start_row = arr[i]
        end_row = arr[i + 1]
        
        # Generate interpolated values between start_row and end_row
        interpolated_steps = np.linspace(start_row, end_row, num=k + 2)  # Including start and end
        interpolated_result.append(interpolated_steps[:-1])  # Exclude the last row to avoid duplication
    
    interpolated_result.append([arr[-1]])  # Add the last row
    return np.vstack(interpolated_result)

num_trajectories = 30

for k in range (1, 4):
    for i in range(1, num_trajectories+1):
        if i < 10:
            trajectory_number = '0'+str(i)
        else:
            trajectory_number = str(i)

        # Trajectory 7 was corrupted
        if i == 7:
            continue

        np_action = np.load('data/aloha_hand_over_reduced_interpol/trajectory_'+trajectory_number+'/action.npz')
        np_action = np_action['arr_0']
        np_position = np.load('data/aloha_hand_over_reduced_interpol/trajectory_'+trajectory_number+'/agent_pos.npz')
        np_position = np_position['arr_0']
        np_velocity = np.load('data/aloha_hand_over_reduced_interpol/trajectory_'+trajectory_number+'/agent_vel.npz')
        np_velocity= np_velocity['arr_0']


        np_action = interpolate_between_all_rows(np_action, k)
        # print('np_action')
        # print(np_action.shape)
        # np_action = np_action[:, :-1]

        np_position = interpolate_between_all_rows(np_position, k)
        # print('np_position')
        # print(np_position.shape)
        # np_position = np_position[:, :-2]

        np_velocity = interpolate_between_all_rows(np_velocity, k)
        # print('np_velocity')
        # print(np_velocity.shape)
        # np_velocity = np_velocity[:, :-2]

        if i < 10 and k == 0:
            new_trajectory_number = '0'+str(i+num_trajectories*(k))
        else:
            new_trajectory_number = str(i+num_trajectories*(k))

        output_dir = os.path.join('data/aloha_hand_over_reduced_interpol/trajectory_'+new_trajectory_number)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        np.savez('data/aloha_hand_over_reduced_interpol/trajectory_'+new_trajectory_number+'/action' , np_action)
        # np_action = np.load('data/aloha_hand_over/trajectory_'+trajectory_number+'/action.npz')
        # np_action = np_action['arr_0']
        # print('np_action')
        # print(np_action.shape)

        np.savez('data/aloha_hand_over_reduced_interpol/trajectory_'+new_trajectory_number+'/agent_pos' , np_position)
        # np_position = np.load('data/aloha_hand_over/trajectory_'+trajectory_number+'/agent_pos.npz')
        # np_position = np_position['arr_0']
        # print('np_position')
        # print(np_position.shape)


        np.savez('data/aloha_hand_over_reduced_interpol/trajectory_'+new_trajectory_number+'/agent_vel' , np_velocity)
        # np_velocity = np.load('data/aloha_hand_over/trajectory_'+trajectory_number+'/agent_vel.npz')
        # np_velocity = np_velocity['arr_0']
        # print('np_velocity')
        # print(np_velocity.shape)