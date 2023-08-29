import numpy as np
import os
import shutil

img_path = "/media/dheeraj/New_Volume/Waterloo-Work/Research_Work/bowl-particles-tracking/detection-based-tracking/deep_sort/BOWL_DC/train/BOWL_DC/img1/"

new_path = "/media/dheeraj/New_Volume/Waterloo-Work/Research_Work/bowl-particles-tracking/detection-based-tracking/deep_sort/BOWL_DC/train/BOWL_DC/hello/"

if not os.path.exists(new_path):
    os.makedirs(new_path)
images = sorted(os.listdir(img_path))

print(images)
count = 0
for i in images:
    img_name = os.path.join(img_path, i)
    new_name = '{:05d}'.format(count) + ".txt"
    print(img_name)
    new2_path = os.path.join(new_path, new_name)
    print(new2_path)
    shutil.copy2(img_name, new2_path)
    count +=1

