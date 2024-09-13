import numpy as np

for i in range(1, 31):
# for i in range(1, 2):
    if i < 10:
        trajectory_number = '0'+str(i)
    else:
        trajectory_number = str(i)

    if i == 7:
        continue

    np_action = np.load('data/aloha_hand_over/trajectory_'+trajectory_number+'/action.npz')
    np_action = np_action['arr_0']
    np_position = np.load('data/aloha_hand_over/trajectory_'+trajectory_number+'/agent_pos.npz')
    np_position = np_position['arr_0']
    np_velocity = np.load('data/aloha_hand_over/trajectory_'+trajectory_number+'/agent_vel.npz')
    np_velocity= np_velocity['arr_0']

    print('np_action')
    print(np_action.shape)
    np_action = np_action[:, :-1]

    print('np_position')
    print(np_position.shape)
    np_position = np_position[:, :-2]

    print('np_velocity')
    print(np_velocity.shape)
    np_velocity = np_velocity[:, :-2]

    np.savez('data/aloha_hand_over/trajectory_'+trajectory_number+'/action' , np_action)
    np_action = np.load('data/aloha_hand_over/trajectory_'+trajectory_number+'/action.npz')
    np_action = np_action['arr_0']
    print('np_action')
    print(np_action.shape)

    np.savez('data/aloha_hand_over/trajectory_'+trajectory_number+'/agent_pos' , np_position)
    np_position = np.load('data/aloha_hand_over/trajectory_'+trajectory_number+'/agent_pos.npz')
    np_position = np_position['arr_0']
    print('np_position')
    print(np_position.shape)


    np.savez('data/aloha_hand_over/trajectory_'+trajectory_number+'/agent_vel' , np_position)
    np_velocity = np.load('data/aloha_hand_over/trajectory_'+trajectory_number+'/agent_vel.npz')
    np_velocity = np_velocity['arr_0']
    print('np_velocity')
    print(np_velocity.shape)