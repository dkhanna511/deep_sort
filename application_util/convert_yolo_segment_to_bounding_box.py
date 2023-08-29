import os
import csv
import numpy as np
from itertools import repeat
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

directory = "/media/dheeraj/New_Volume/Waterloo-Work/Research_Work/bowl-particles-tracking/detection-based-tracking/deep_sort/BOWL_TITAN_segment_res/TRACK_TITANSTART0008/newest_labels/"
image_width = 672
image_height = 512
file = "/media/dheeraj/New_Volume/Waterloo-Work/Research_Work/bowl-particles-tracking/detection-based-tracking/deep_sort/BOWL_TITAN_segment_res/train/BOWL_TITAN/det/det.txt"

def get_bounding_box(points):

    minx, miny = 10000, 10000
    maxx, maxy = 0,0
    for x, y in points:
        # Set min coords
        x, y = float(x), float(y)
        # print("x is : {}, y is : {}".format(type(x), type(y)))
        if float(x) < minx:
            minx = float(x)
        if float(y) < miny:
            miny = float(y)
        # Set max coords
        if float(x) > maxx:
            maxx = x
        if float(y) > maxy:
            maxy = y


    return minx, miny, maxx- minx, maxy-miny
with open(file, 'w') as output:
    write = csv.writer(output, delimiter=',')
    for index, i in enumerate(sorted(os.listdir(directory))):
        print(index)
        file_name = os.path.join(directory, i)
        print(file_name)
        with open(file_name, 'r') as f:
            for lines in f:
                line = []
                val = []
                # print(lines)
                lines = lines.split(" ")[1:-1]
                polygon = []
                # print("lines are : ", lines)
                for i in range(0, len(lines)-1, 2):
                    polygon.append((lines[i], lines[i+1]))
                    
                if len(polygon) > 3:
                    minx, miny, box_width, box_height = get_bounding_box(polygon)
                else:
                    continue
                # print(" polygon is : ", polygon)
                # print(" minx {} miny {} maxx {} maxy {}".format(minx, miny, box_width, box_height))
                # box = polygon.minimum_rotated_rectangle
                # if minx == 10000:
                #     print("here")
                #     print("lines are : ", lines)
                #     # exit(0)
                #     continue
                line.append(int(index))
                line.append(-1)
                # lines.extenlinesd(repeat(8, 2))
                # print(" lines is : ", lines)
                line.append(minx * image_width)
                line.append(miny * image_height)
                
                line.append(image_width * box_width)
                line.append(image_height * box_height)
               
                line.extend(repeat(1, 3))
                
                # print(" lines is : ", line)
                print(" line is : ", line)
                val.append(line)
                
                write.writerows(val)



with open(file, 'r') as input:
    lines = input.readlines()

data = np.array([line.strip().split(",") for line in lines], dtype=np.float)
np.savetxt('BOWL_20.txt', data)


            # print(lines)
        # line = lines.split("\n")
        # for j in lines:

        #     print(j)
        #     line = j.split(" ")
        #     print(line[1:])
        # # print(line)
        # break
 