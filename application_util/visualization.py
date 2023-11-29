# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys
import cv2
from .image_viewer import ImageViewer
import json
import configs
from shapely.geometry import Point, mapping
from shapely.geometry.polygon import Polygon
def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, seq_info, update_ms):
        image_shape = seq_info["image_size"][::-1]
        print(" image shape is : ", image_shape)
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)
        # print(" image shape is : ", image_shape)

        # exit(0)
        
        self.viewer = ImageViewer(
            update_ms, image_shape, "Figure %s" % seq_info["sequence_name"])
        self.viewer.thickness = 1
        self.line_color = (0, 255, 0)
        self.pixel_step = 100
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        return True
    
    # def get_top_left(self, polygon):
        
    def set_image(self, image):
        self.viewer.image = image
        # x = self.pixel_step
        # y = self.pixel_step
        # grid_xs = np.linspace(0, 400, num = 5)
        # grid_ys = np.linspace(0, 400, num = 5)
        # count = 0
        # dictionary = {}
        # for i in grid_xs:
        #     for j in grid_ys:
        #         # print(count)
        #         # strrr = str(i) + "," + str(j)
        #         cv2.putText(image, str(count), (int(i) +  x/2, int(j)+ y/2), cv2.FONT_HERSHEY_PLAIN,2, (0, 255, 0),1 )
        #         count +=1
                
        # while x < image.shape[1]:
        #     cv2.line(image, (x, 0), (x, image.shape[0]), color=self.line_color, lineType=cv2.LINE_AA, thickness=self.viewer.thickness)
        #     x += self.pixel_step

        # while y < image.shape[0]:
        #     cv2.line(image, (0, y), (image.shape[1], y), color=self.line_color, lineType=cv2.LINE_AA, thickness=self.viewer.thickness)
        #     y += self.pixel_step


        if configs.bowl_name =='titan':
            json_file_path = "./Bowl_annotations/bowl_titan_latest_normal_operation.json"
        elif configs.bowl_name =='20':
            json_file_path = "./Bowl_annotations/bowl_data_15_20_titan.json"
        # print("annotation json")
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
        x_poly, y_poly = [], []
        points = []
        for i in range(0, len(data['bowl_{}'.format(configs.bowl_name)]['regions'])):
            x_points =  data['bowl_{}'.format(configs.bowl_name)]['regions'][i]['shape_attributes']['all_points_x']
            y_points = data['bowl_{}'.format(configs.bowl_name)]['regions'][i]['shape_attributes']['all_points_y']
            # x_poly.append(x_points)
            # y_poly.append(y_points)
            polygon = []
            for j in range(len(x_points)):
                polygon.append([x_points[j], y_points[j]])
            # print(" polygon is :", polygon)
            points.append(polygon)

       
        # Attributes
        isClosed = True
        color = (255, 0, 0)
        thickness = 2
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(points) - 1, 3),
		            dtype="uint8")
        COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

        legend = np.zeros(((len(points) * 25) + 25, 300, 3), dtype="uint8")
        # draw closed polyline
        for i in range (0, len(points)):
            # color = COLORS[i]
            # cv2.putText(legend, i, (5, (i * 25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	        # cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25),tuple(color), -1)
            polygon = Polygon(points[i])
            xx,yy,w,h = cv2.boundingRect(np.array(points[i]))
            rect = cv2.minAreaRect(np.array(points[i]))
            box = cv2.boxPoints(rect)
            # print(" box is : ", box)
            # exit(0)
            pts = np.array(points[i], dtype=np.int32)
            # print("point[i] is : ", points[i])
            # topmost_then_leftmost_point =  min(pts, key=lambda pt: (pt[0][1],  pt[0][0]))[0]
            min_x = min(p[0] for p in pts)
            min_y = min(p[1] for p in pts)
            # print(" top left point is : ", topmost_then_leftmost_point)
            centroid = mapping(polygon.centroid)
            xx, yy = centroid['coordinates']
            # exit(0)
            
            # print("pts are :", pts)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed, color, thickness)

            # cv2.putText(image, str(i), (int(xx)-15, int(yy)), cv2.FONT_HERSHEY_PLAIN,1, (255, 255, 0),2 )

            
        cv2.putText(image, "image dim: {}, {}".format(image.shape[1], image.shape[0]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 0),2 )

            
    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        # for track_id, box in zip(track_ids, boxes):
            # self.viewer.color = create_unique_color_uchar(track_id)
            # self.viewer.rectangle(*box.astype(np.int), label=str(track_id))
# 
    def draw_detections(self, detections):
        self.viewer.thickness = 1
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            # print("detection is :",detection.tlwh)
            self.viewer.rectangle(*detection.tlwh)

    def draw_velocities(self, region_vel, points):
        self.viewer.thinkness = 1
        self.viewer.color = 0, 0, 0
        self.viewer.velocity(region_vel, points)
        # for keys in curr_frame_polygons:
        

    def draw_trackers(self, tracks):
        self.viewer.thickness = 1
        # for track in tracks:
        #     if not track.is_confirmed() or track.time_since_update > 0:
        #         continue
        #     self.viewer.color = create_unique_color_uchar(track.track_id)
        #     self.viewer.rectangle(
        #         *track.to_tlwh().astype(np.int), label=str(track.track_id))
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)
#
