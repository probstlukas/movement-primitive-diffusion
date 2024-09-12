import numpy as np

# data = np.load("data/obstacle_avoidance/trajectroy_00/joint_pos.npz")
data = np.load("data/obstacle_avoidance/trajectroy_00/observation.npz")
lst = data.files
print(data['arr_0'])
# for item in lst:
#     print(item)
#     print(data[item])