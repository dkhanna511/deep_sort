import numpy as np
import os
import shutil

label_path = "/media/dheeraj/New_Volume/Waterloo-Work/Research_Work/bowl-particles-tracking/detection-based-tracking/deep_sort/BOWL_TITAN_segment_res/TRACK_TITANSTART0008/labels"

new_path = "/media/dheeraj/New_Volume/Waterloo-Work/Research_Work/bowl-particles-tracking/detection-based-tracking/deep_sort/BOWL_TITAN_segment_res/TRACK_TITANSTART0008/newest_labels"

if not os.path.exists(new_path):
    os.makedirs(new_path)
labels = sorted(os.listdir(label_path))

print(label_path)
# count = 0
for i in labels:
    label_name = os.path.join(label_path, i)
    val = int(i.split("_")[1].split(".")[0])
    print(i)
    # print(val)
    new_name = '{:05d}'.format(val) + ".txt"
    print(new_name)
    new2_path = os.path.join(new_path, new_name)
    print(new2_path)
    shutil.copy2(label_name, new2_path)
    # count +=1

