# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import time
import json
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

json_file_path = "./bowl_data_15_20_titan.json"
with open(json_file_path, "r") as json_file:
    data = json.load(json_file)
points = []
for i in range(1, len(data['bowl_20']['regions'])):
    x_points =  data['bowl_20']['regions'][i]['shape_attributes']['all_points_x']
    y_points = data['bowl_20']['regions'][i]['shape_attributes']['all_points_y']
    # x_poly.append(x_points)
    # y_poly.append(y_points)
    polygon = []
    for j in range(len(x_points)):
        polygon.append((x_points[j], y_points[j]))
    # print("opolygon is : ", polygon)
    # polygon = Polygon(polygon)
    # print(" polygon is :", polygon)
    
    points.append(polygon)

# exit(0)

def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    # print("frame indices is : ", frame_indices)
    # exit(0)
    mask = frame_indices == frame_idx
    # print("mask indices is : ", mask[:175])
    # exit(0)
    detection_list = []
    # print(" length of detection[mask] is : ", len(detection_mat[mask]))
    # exit(0)
    for row in detection_mat[mask]:
        # print("row is ", len(row))
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        # print(" shape of bounding box : {}, confidence is : {}, feature is : {}".format(bbox.shape, confidence.shape, feature.shape))
        # print("bbox is : ", bbox)
        
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def get_velocity_data_structure(points):
    # for i in range(len(points)):
        # print(points[i])
    pairs = {}
    for i in range(len(points)):
        pairs[i] = {"length" : len(points[i]), "points": points[i], "objects_encountered" : [],"object_positions" : [], "object_time": None}
        
    # print(pairs)
    return pairs
    

def get_polygons(points):
    for i in range(len(points)):
        print(points[i])
    pairs = {}
    for i in range(len(points)):
        min_x, min_y, max_x, max_y = 10000, 10000, 0, 0
        for tuple in points[i]:
            # print("points[i]", tuple)
            # exit(0)
            if tuple[0] < min_x:
                min_x = tuple[0]
            if tuple[1] < min_y:
                min_y = tuple[1]
            if tuple[0] > max_x:
                max_x = tuple[0]
            if tuple[1] > max_y:
                max_y = tuple[0]
        if max_x - min_x > max_y - min_y:
            print(" maxx and minx are : {}, {}".format(max_x, min_x))
            pairs[i] = {"length" : len(points[i]), "points": points[i], "max_distance" : max_x - min_x, "objects_encountered" : [],}
        
        if max_y - min_y > max_x - min_x:
            pairs[i] = {"length" : len(points[i]), "points": points[i], "max_distance" : max_y - min_y, "objects_encountered" : []}
    print(pairs)
    return pairs
    
# exit(0)
global prev_frame_stats
prev_frame_stats = {}

def get_empty_pair_grid(xs, ys):
    count = 0
    pairs = {}
    grid_width, grid_height = 100, 100

    for i in xs:
        for j in ys:
            pairs[count] = ([(i, j), (i + grid_width, j + grid_height)])
            count +=1

    return pairs



def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    print("seq info is : ", seq_info)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    print("metric : ", metric)
    tracker = Tracker(metric)
    results = []
    pairs = {}
    # print("pairs are : ", pairs)
    # exit(0)
    xs = np.linspace(0, 400, num = 5)
    ys = np.linspace(0, 400, num = 5)
    # pairs = get_empty_pair_grid(xs, ys)
    pairs = get_polygons(points)

    global counter
    counter = 0

    instantaneous_velocity = {}


    def frame_stats(pairs, results, points):

        regions = get_velocity_data_structure(points)
        for keys in pairs:
            object_id_list = []
            object_position_list = []
            object_id_time = 0
            
            for j in range(len(results)):
                point_tl, point_br = Point(results[j][2], results[j][3]), Point(results[j][2] + results[j][4], results[j][3]+results[j][5]) 
                region = Polygon(pairs[keys]["points"])
                
                # if (results[j][2] > pairs[i][0][0]) and (results[j][3] >  pairs[i][0][1]) and  (results[j][2] + results[j][4] < pairs[i][1][0]) and (results[j][3] + results[j][5] < pairs[i][1][1]):
                if region.contains(point_tl) and region.contains(point_br):
                    object_id_list.append(results[j][1])
                    object_position_list.append((results[j][2]+results[j][4]/2, results[j][3] + results[j][5]/2))
                    object_id_time = time.time()
        
            if len(object_id_list) > 0:
                regions[keys]["objects_encountered"].extend(object_id_list)
                regions[keys]["object_positions"].extend(object_position_list)
                regions[keys]["object_time"] = object_id_time
                # print(regions[keys])
                # continue
        # exit(0)

        # print("regions are : ",regions)
        return regions

        
    def get_region_velocities(prev_frame_stats, curr_frame_stats):
        

        # print(" previous frame stats : ", prev_frame_stats)
        # print("curfrent frame stats : ", curr_frame_stats)
        region_vel = {}

        for keys in curr_frame_stats:
            region_vel[keys] = {"velocity" : 0.0}

        for keys in curr_frame_stats:
            # print(keys)
            # print(" previous frames encountered : ", prev_frame_stats[keys]["objects_encountered"])
            
            
            list1_array = np.array(prev_frame_stats[keys]["objects_encountered"])
            list2_array = np.array(curr_frame_stats[keys]["objects_encountered"])
            matching_indices = np.where(np.isin(list1_array, list2_array))[0]
            matching_ids = list1_array[matching_indices]
            # matching_ids_copy = prev_frame_stats[matching_indices]
            # indexes = []
            # print(" matching ids are {} and {}".format(matching_ids, matching_ids_copy))
            # matching_id_copy  = matching_ids.tolist()
            part_instant_velocity = []
            for index, num in enumerate(matching_ids):
                # print(" num is : ", num)
                for i, value in enumerate(curr_frame_stats[keys]["objects_encountered"]):
                    # print(" value is : {} and i is : {}".format(value, i))
                    if value == num:
                        # indexes.append(i)
                        # print(" here")
                        id_displacement = np.sqrt((curr_frame_stats[keys]["object_positions"][i][0] - prev_frame_stats[keys]["object_positions"][index][0])**2 + 
                                                  (curr_frame_stats[keys]["object_positions"][i][1] - prev_frame_stats[keys]["object_positions"][index][1])**2)
                        time_taken = curr_frame_stats[keys]["object_time"] - prev_frame_stats[keys]["object_time"]
                        part_instant_velocity.append(id_displacement / time_taken)
                        break
                    
            region_vel[keys]["velocity"]  =round(np.mean(part_instant_velocity), 4)
            if np.isnan(region_vel[keys]["velocity"]):
                region_vel[keys]["velocity"] = 0.0
                    
        
        print(" region velocity is : ", region_vel)

        return region_vel

            
            



    def get_jammed_widgets(pairs, results):
        
        # mean_square_list = []
        # print(" length of results is : ", len(results))
        # exit(0)
        # frames_grid_before = []
        for keys in pairs:
                # print(" x is : {} and y is : {}".format(pairs[i][0], pairs[i][1]))
            # print(" keys are :", keys)
            frames_grid_after = []
            for j in range(len(results)):
                point_tl, point_br = Point(results[j][2], results[j][3]), Point(results[j][2] + results[j][4], results[j][3]+results[j][5]) 
                region = Polygon(pairs[keys]["points"])
                
                
                # if (results[j][2] > pairs[i][0][0]) and (results[j][3] >  pairs[i][0][1]) and  (results[j][2] + results[j][4] < pairs[i][1][0]) and (results[j][3] + results[j][5] < pairs[i][1][1]):
                if region.contains(point_tl) and region.contains(point_br):
                    frames_grid_after.append(results[j][1])
                # print(" i is :",i)
                # exit(0)
                # print(" pair[i] is : ", pairs[i])
                    
            if len(frames_grid_after) > 0:  
                # print(" yeah reaching here and this is working fine")   
                # print(pairs[keys]["objects_encountered"])
                pairs[keys]["objects_encountered"].append(frames_grid_after)    
            # pairs[i].append("hello")
            
            # if len(frames_grid_before) == 0:
            #     frames_grid_before = frames_grid_after
            # frames_grid_before_np = np.asarray(frames_grid_before)
            frames_grid_after_np = np.asarray(frames_grid_after)

                        # print(" length of pairs is :", pairs[keys]["objects_encountered"])
            if len(pairs[keys]["objects_encountered"]) > 100:
                # print(" prev objects are : ", pairs[i][len(pairs[i])-27])
                # print(" current objects are : ", frames_grid_after)
                matching_ids = np.intersect1d(pairs[keys]["objects_encountered"], frames_grid_after_np)
                # print("matching ids are :", matching_ids)
                # exit(0)
                # pairs[i-8] = []
                # pairs = get_empty_pair_grid(xs, ys)
                if len(matching_ids) > 0:
                    # print("matching ids are : ", matching_ids)
                    if len(matching_ids)/len(frames_grid_after_np)> 0.9:
                        print(" {} widgets jammed at grid : {}".format(len(matching_ids), keys+1))
                        # time.sleep(1)
            # frames_grid_before = frames_grid_after
            
        # print(" pair is : ", pairs)
        # time.sleep(2)
            

            # print(" pair is : {}, and matching ids are : {}".format(pairs[i], frames_grid_after[matching_ids]))
    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)
        frames_grid_before = []
        global prev_frame_stats
        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]
        # print(" length of detections is : ", len(detections))
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # print(" detections are :", detections)
        # Update tracker.
        tracker.predict()
        tracker.update(detections)
        
        # print("shape of tracker is :", len(tracker.tracks))
        # exit(0)
        # get_mean_squared_error(pairs, detections, frame_idx)
        # Update visualization.
        
        # Store results.
        temp_res = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            temp_res.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
            results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
        # print("result is :", results)
        # get_jammed_widgets(pairs, temp_res)
        # get_region_velocities(pairs, temp_res)
        
        if len(prev_frame_stats)==0:
            prev_frame_stats = frame_stats(pairs, temp_res, points)
            region_vel = {}
        else:
            curr_frame_stats = frame_stats(pairs, temp_res, points)
            region_vel = get_region_velocities(prev_frame_stats, curr_frame_stats)
            prev_frame_stats = curr_frame_stats
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            if region_vel !={} :
                vis.draw_velocities(region_vel, points)
            vis.draw_trackers(tracker.tracks)

        # exit(0)
    # Run tracker.output_file
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=1)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")
def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.5)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
