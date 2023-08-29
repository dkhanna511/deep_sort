import os
import csv
import numpy as np
from itertools import repeat


directory = "/media/dheeraj/New_Volume/Waterloo-Work/Research_Work/bowl-particles-tracking/detection-based-tracking/deep_sort/BOWL_20/predict0200/newest_labels"
image_width = 672
image_height = 512
file = "/media/dheeraj/New_Volume/Waterloo-Work/Research_Work/bowl-particles-tracking/detection-based-tracking/deep_sort/BOWL_20/train/BOWL_20/det/det.txt"
with open(file, 'w') as output:
    write = csv.writer(output, delimiter=',')
    for index, i in enumerate(sorted(os.listdir(directory))):
        print(index)
        file_name = os.path.join(directory, i)
        print(file_name)
        with open(file_name, 'r') as f:
            for lines in f:
                val = []
                # print(lines)
                lines = lines.split(" ")
                lines = lines[1:5]
                lines.insert(0, int(index))
                lines.insert(1, int(-1))
                # lines.extenlinesd(repeat(8, 2))
                lines.extend(repeat(1, 3))
                # print(" lines is : ", lines)
                
                
                lines[4] = image_width * float(lines[4])
                lines[5] = (image_height * float(lines[5]))
                lines[2] = round(image_width * float(lines[2]) - lines[4]/2)
                lines[3] = round(image_height * float(lines[3]) - lines[5]/2)
                print(" lines is : ", lines)
                
                val.append(lines)
                

                # lines
                # print(lines)

                # exit(0)
                # output.write(lines)
                # val = np.asarray(val)
                write.writerows(val)



with open(file, 'r') as input:
    lines = input.readlines()

data = np.array([line.strip().split(",") for line in lines], dtype=np.float)
np.savetxt('BOWL_20_NP.txt', data)


            # print(lines)
        # line = lines.split("\n")
        # for j in lines:

        #     print(j)
        #     line = j.split(" ")
        #     print(line[1:])
        # # print(line)
        # break
 