import sys
sys.path.insert(0, "../torchdriveenv")

import os
import pickle
import shutil

data_dir = "data/waypoint_graph_no_rendering_test_tiny"
dst_dir = "data/selected_more_waypoints_val"
file_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

#for i in range(100):
#    file_path = random.choice(file_paths)

for file_path in file_paths:
    with open(file_path, "rb") as f:
        episode_data = pickle.load(f)
#     print(episode_data)
    if episode_data.step_data[-1].info["reached_waypoint_num"] > 2:
        print(episode_data.step_data[-1].info["reached_waypoint_num"], ' ', file_path)
        shutil.copy(file_path, dst_dir)
