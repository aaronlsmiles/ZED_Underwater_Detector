#!/usr/bin/env python3

import sys
import numpy as np
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from threading import Lock, Thread
from time import sleep
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import pandas as pd
from datetime import datetime
import json  # Add this import
import os

lock = Lock()
run_signal = False
exit_signal = False
pause_signal = False
class_names = None  # Add this line for storing YOLO class names
detections = None
image_net = None
zed = None
excel_filename = None
svo_recording = None
current_objects = None
run_folder = None


def start_svo_recording(run_folder, timestamp):
    global svo_recording, excel_filename

    svo_path = os.path.join(run_folder, f"recording_{timestamp}.svo")

    recording_param = sl.RecordingParameters(svo_path, sl.SVO_COMPRESSION_MODE.H264_LOSSLESS)
    err = zed.enable_recording(recording_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error enabling recording: {err}")
    else:
        print(f"Started recording to {svo_path}")

def save_screenshot(image, timestamp, point_cloud, object_distance, width, height, depth,
                    upper_distance, center_distance, lower_distance, current_time,
                    det, class_names):  # Add these parameters
    """Save a screenshot with a timestamp-based filename and center point markers"""
    # Remove global variables - we'll pass what we need instead

    # Create a copy of the image to draw on
    display_image = image.copy()

    # Calculate center coordinates
    bbox = det.bounding_box_2d
    center_x = int((bbox[0][0] + bbox[2][0]) * 0.5)
    center_y = int((bbox[0][1] + bbox[2][1]) * 0.5)

    # Calculate upper and lower points
    upper_y = center_y - 50
    lower_y = center_y + 50

    # Ensure coordinates are within image bounds
    height, width = display_image.shape[:2]
    center_x = max(0, min(center_x, width - 1))
    center_y = max(0, min(center_y, height - 1))
    upper_y = max(0, min(upper_y, height - 1))
    lower_y = max(0, min(lower_y, height - 1))

    color = (255, 0, 0)  # Red color (RGB format)
    size = 20
    thickness = 3

    # Draw markers for all three points
    points = [
        ("Upper", center_x, upper_y, (center_x + 25, upper_y + 5), upper_distance),
        ("Center", center_x, center_y, (center_x + 25, center_y + 5), center_distance),
        ("Lower", center_x, lower_y, (center_x + 25, lower_y + 5), lower_distance)
    ]

    for point_name, x, y, text_pos, distance in points:
        # Draw cross
        cv2.line(display_image, (x - size, y), (x + size, y), color, thickness)
        cv2.line(display_image, (x, y - size), (x, y + size), color, thickness)
        cv2.circle(display_image, (x, y), 5, color, -1)

        # Display distance
        distance_text = f"{distance:.1f}mm"
        cv2.putText(display_image, distance_text, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Update the detection data
    new_row = pd.DataFrame({
        'Class': [det.label],
        'Name': [class_names[det.label]],
        'Confidence': [f"{det.probability:.2%}"],
        'Upper_Distance': [f"{upper_distance:.1f}" if isinstance(upper_distance, float) else "N/A"],
        'Center_Distance': [f"{center_distance:.1f}" if isinstance(center_distance, float) else "N/A"],
        'Lower_Distance': [f"{lower_distance:.1f}" if isinstance(lower_distance, float) else "N/A"],
        'Object_Distance': [str(object_distance)],
        'Width': [str(width)],
        'Height': [str(height)],
        'Depth': [str(depth)],
        'Timestamp': [current_time]
    })
    global detection_data
    detection_data = pd.concat([detection_data, new_row], ignore_index=True)

    filename = os.path.join(run_folder, f"screenshot_{timestamp}.png")
    cv2.imwrite(filename, cv2.cvtColor(display_image, cv2.COLOR_RGBA2BGR))
    print(f"Screenshot saved as {filename}")


def parse_timestamp(timestamp_str):
    """Convert timestamp string to datetime object"""
    if '---PAUSED---' in timestamp_str:
        # Extract the timestamp from the screenshot filename
        timestamp_str = timestamp_str.split('screenshot_')[1].split('.')[0]
        return datetime.strptime(timestamp_str, "%d%m%Y_%H%M%S")
    else:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")


def is_in_pause_period(current_time_dt, pause_periods):
    """Check if current time falls within any pause period"""
    for pause_start, pause_end in pause_periods:
        pause_start_dt = parse_timestamp(pause_start)
        pause_end_dt = parse_timestamp(pause_end) if pause_end else None

        if pause_end_dt is None:
            # For open-ended pause periods, check if we're after the start
            if current_time_dt >= pause_start_dt:
                return True
        else:
            # For closed pause periods, check if we're within the range
            if pause_start_dt <= current_time_dt <= pause_end_dt:
                return True
    return False

def generate_run_folder_name():
    # Get current timestamp
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

    # Create base name with timestamp
    folder_name = f"run_{timestamp}"

    # Add SVO name if in playback mode
    if opt.svo:
        svo_name = os.path.basename(opt.svo).split('.')[0]
        folder_name += f"_{svo_name}"

    # Add enhancement flags
    enhancements = []
    if opt.use_fakemix:
        enhancements.append("fakemix")
    if opt.use_fusion:
        enhancements.append("fusion")
    if opt.use_edge:
        enhancements.append("edge")
    if opt.use_depth:
        enhancements.append("depth")
    if opt.use_multiview:
        enhancements.append("multiview")
    if opt.use_temporal:
        enhancements.append("temporal")

    if enhancements:
        folder_name += "_" + "_".join(enhancements)

    return folder_name, timestamp


def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) #* im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) #* im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) #* im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) #* im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output

def detections_to_custom_box(detections, im0):
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
        # Convert numpy array to integer properly
        class_id = int(det.cls.item())  # Extract scalar value from numpy array
        obj.label = class_id  # Add this line to set the label
        obj.probability = float(det.conf.item())  # Convert confidence to float
        obj.is_grounded = False
        output.append(obj)
    return output


class EnhancedFusionModule:
    def __init__(self):
        self.edge_detector = cv2.createEdgeDetector()

    def extract_features(self, image):
        # Extract edge features
        edges = cv2.Canny(image, 100, 200)
        edges = cv2.dilate(edges, None, iterations=2)

        # Extract depth features
        depth = sl.Mat()
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        depth_map = depth.get_data()

        # Extract color features
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        return edges, depth_map, hsv

    def fuse_features(self, edges, depth_map, hsv):
        # Normalize features
        edges = edges.astype(np.float32) / 255.0
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

        # Create feature fusion
        fused = np.zeros_like(edges)

        # Weight the features
        edge_weight = 0.4
        depth_weight = 0.3
        color_weight = 0.3

        # Combine features
        fused = (edge_weight * edges +
                 depth_weight * depth_map +
                 color_weight * cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

        return fused


def fake_mix_augmentation(image, alpha=0.5):
    """Implements FakeMix augmentation for transparent objects"""
    mixed_image = image.copy()
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    edges = cv2.Canny(image, 100, 200)
    edges = cv2.dilate(edges, None, iterations=2)
    mask = edges > 0
    mixed_image[mask] = cv2.addWeighted(
        mixed_image[mask],
        1 - alpha,
        noise[mask],
        alpha,
        0
    )
    return mixed_image


def merge_detections(det1, det2):
    """Merge detections from multiple views"""
    if det1 is None:
        return det2
    if det2 is None:
        return det1

    # Combine detections
    combined = np.concatenate([det1, det2])

    # Apply non-maximum suppression
    # You might want to implement a more sophisticated merging strategy
    return combined


class DetectionEnhancer:
    def __init__(self, use_fakemix=False, use_fusion=False, use_edge=False,
                 use_depth=False, use_multiview=False, use_temporal=False):
        self.use_fakemix = use_fakemix
        self.use_fusion = use_fusion
        self.use_edge = use_edge
        self.use_depth = use_depth
        self.use_multiview = use_multiview
        self.use_temporal = use_temporal

        if self.use_fusion:
            self.fusion_module = EnhancedFusionModule()

        if self.use_temporal:
            self.previous_detections = []
            self.temporal_threshold = 0.3

    def enhance_image(self, image, zed=None):
        enhanced_image = image.copy()

        if self.use_fusion:
            edges, depth_map, hsv = self.fusion_module.extract_features(image)
            enhanced_image = self.fusion_module.fuse_features(edges, depth_map, hsv)

        if self.use_edge:
            edges = cv2.Canny(enhanced_image, 100, 200)
            enhanced_image = cv2.addWeighted(enhanced_image, 0.7,
                                             cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.3, 0)

        if self.use_fakemix:
            enhanced_image = fake_mix_augmentation(enhanced_image)

        return enhanced_image

    def apply_temporal_consistency(self, current_detections):
        if not self.use_temporal or not self.previous_detections:
            self.previous_detections = current_detections
            return current_detections

        filtered_detections = []
        for det in current_detections:
            for prev_det in self.previous_detections:
                if (det.cls == prev_det.cls and
                        abs(det.xywh[0][0] - prev_det.xywh[0][0]) < self.temporal_threshold and
                        abs(det.xywh[0][1] - prev_det.xywh[0][1]) < self.temporal_threshold):
                    filtered_detections.append(det)
                    break

        self.previous_detections = current_detections
        return filtered_detections

    def adjust_confidence(self, detection, image, depth_map=None):
        if not (self.use_depth or self.use_fusion):
            return detection

        center_x = int(detection.xywh[0][0])
        center_y = int(detection.xywh[0][1])

        confidence_adjustment = 1.0

        if self.use_depth and depth_map is not None:
            depth_value = depth_map[center_y, center_x]
            depth_factor = 1.0 / (1.0 + depth_value / 1000.0)
            confidence_adjustment *= depth_factor

        if self.use_fusion:
            edges, _, _ = self.fusion_module.extract_features(image)
            edge_value = edges[center_y, center_x]
            edge_factor = edge_value / 255.0
            confidence_adjustment *= (0.6 + 0.4 * edge_factor)

        # Create a new detection with adjusted confidence
        # Use numpy's copy instead of PyTorch's clone
        new_data = detection.data.copy()
        new_data[0, 4] = detection.conf * confidence_adjustment  # Update confidence in the data array
        new_detection = type(detection)(new_data, detection.orig_shape)
        return new_detection


def save_enhancement_config(timestamp, enhancements, run_folder):
    """Save enhancement configuration to a JSON file"""
    config = {
        'timestamp': timestamp,
        'enhancements': {
            'use_fakemix': enhancements['use_fakemix'],
            'use_fusion': enhancements['use_fusion'],
            'use_edge': enhancements['use_edge'],
            'use_depth': enhancements['use_depth'],
            'use_multiview': enhancements['use_multiview'],
            'use_temporal': enhancements['use_temporal']
        }
    }

    config_filename = os.path.join(run_folder, f"config_{timestamp}.json")
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Enhancement configuration saved to {config_filename}")
    return config_filename

def load_enhancement_config(config_file):
    """Load enhancement configuration from a JSON file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config['enhancements']



def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections, class_names, zed, detection_data, pause_signal, current_objects

    print("Initializing Network...")
    model = YOLO(weights)
    class_names = model.names

    # Initialize enhancer with command line arguments
    enhancer = DetectionEnhancer(
        use_fakemix=opt.use_fakemix,
        use_fusion=opt.use_fusion,
        use_edge=opt.use_edge,
        use_depth=opt.use_depth,
        use_multiview=opt.use_multiview,
        use_temporal=opt.use_temporal
    )

    print("\nActive Enhancements:")
    print(f"FakeMix: {'Enabled' if enhancer.use_fakemix else 'Disabled'}")
    print(f"Fusion Module: {'Enabled' if enhancer.use_fusion else 'Disabled'}")
    print(f"Edge Detection: {'Enabled' if enhancer.use_edge else 'Disabled'}")
    print(f"Depth-aware Confidence: {'Enabled' if enhancer.use_depth else 'Disabled'}")
    print(f"Multi-view Detection: {'Enabled' if enhancer.use_multiview else 'Disabled'}")
    print(f"Temporal Consistency: {'Enabled' if enhancer.use_temporal else 'Disabled'}\n")

    while not exit_signal:
        if run_signal:
            lock.acquire()
            try:
                img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)

                # Get depth map if needed
                depth_map = None
                if opt.use_depth or opt.use_fusion:
                    depth = sl.Mat()
                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                    depth_map = depth.get_data()

                # Apply selected enhancements
                enhanced_img = enhancer.enhance_image(img, zed)

                # Run detection on enhanced image
                det = model.predict(enhanced_img, save=False, imgsz=img_size,
                                    conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes

                # Get right view detections if multi-view is enabled
                right_det = None
                if opt.use_multiview:
                    image_right_tmp = sl.Mat()
                    zed.retrieve_image(image_right_tmp, sl.VIEW.RIGHT)
                    right_img = cv2.cvtColor(image_right_tmp.get_data(), cv2.COLOR_BGRA2RGB)
                    right_det = model.predict(right_img, save=False, imgsz=img_size,
                                              conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes

                if not pause_signal and len(det) > 0:
                    # Apply temporal consistency if enabled
                    if opt.use_temporal:
                        det = enhancer.apply_temporal_consistency(det)

                    # Merge detections if multi-view is enabled
                    if opt.use_multiview:
                        det = merge_detections(det, right_det)

                    # Adjust confidence if needed
                    for d in det:
                        d = enhancer.adjust_confidence(d, img, depth_map)

                    detections = detections_to_custom_box(det, image_net)
                else:
                    detections = []
            finally:
                lock.release()
            run_signal = False
        sleep(0.01)


def main():
    global image_net, exit_signal, run_signal, detections, class_names, zed, pause_signal, detection_data, excel_filename, current_objects


    # Create run folder in the detections directory
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detections")
    run_folder, timestamp = generate_run_folder_name()
    run_folder = os.path.join(base_path, run_folder)
    os.makedirs(run_folder, exist_ok=True)
    pause_periods = []
    if opt.svo_playback and opt.original_excel:
        print(f"Loading pause periods from {opt.original_excel}")
        original_df = pd.read_excel(opt.original_excel)
        current_pause_start = None

        for idx, row in original_df.iterrows():
            if '---PAUSED---' in str(row['Timestamp']):
                if current_pause_start is None:
                    current_pause_start = row['Timestamp']
                    print(f"Found pause start: {current_pause_start}")
            elif current_pause_start is not None:
                pause_periods.append((current_pause_start, row['Timestamp']))
                print(f"Found pause period: {current_pause_start} to {row['Timestamp']}")
                current_pause_start = None

        # Handle case where last entry is a pause
        if current_pause_start is not None:
            pause_periods.append((current_pause_start, None))
            print(f"Found open pause period starting at: {current_pause_start}")

        print(f"Loaded {len(pause_periods)} pause periods")


    # Initialize both DataFrames
    all_detections_data = pd.DataFrame(columns=['Class', 'Name', 'Confidence',
                                                'Upper_Distance', 'Center_Distance', 'Lower_Distance',
                                                'Object_Distance', 'Width', 'Height', 'Depth', 'Timestamp'])

    filtered_detections_data = pd.DataFrame(columns=['Class', 'Name', 'Confidence',
                                                     'Upper_Distance', 'Center_Distance', 'Lower_Distance',
                                                     'Object_Distance', 'Width', 'Height', 'Depth', 'Timestamp'])

    # If in SVO playback mode: load pause periods from original Excel
    pause_periods = []
    if opt.svo_playback and opt.original_excel:
        original_df = pd.read_excel(opt.original_excel)
        current_pause_start = None

        for idx, row in original_df.iterrows():
            if '---PAUSED---' in str(row['Timestamp']):
                if current_pause_start is None:
                    current_pause_start = row['Timestamp']
            elif current_pause_start is not None:
                pause_periods.append((current_pause_start, row['Timestamp']))
                current_pause_start = None

        # Handle case where last entry is a pause
        if current_pause_start is not None:
            pause_periods.append((current_pause_start, None))

    # Generate timestamp for this session
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

    # Set up filenames for both Excel files in the run folder
    all_detections_filename = os.path.join(run_folder, f"all_detections_{timestamp}.xlsx")
    filtered_detections_filename = os.path.join(run_folder, f"filtered_detections_{timestamp}.xlsx")

    zed = sl.Camera()  # Initialize ZED camera

    # Generate timestamp for this session
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

    # Create enhancements dictionary from command line arguments
    enhancements = {
        'use_fakemix': opt.use_fakemix,
        'use_fusion': opt.use_fusion,
        'use_edge': opt.use_edge,
        'use_depth': opt.use_depth,
        'use_multiview': opt.use_multiview,
        'use_temporal': opt.use_temporal
    }

    # Save enhancement configuration in the run folder
    config_file = save_enhancement_config(timestamp, enhancements, run_folder)

    capture_thread = Thread(target=torch_thread,
                            kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()


    print("Initializing Camera...")

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.optional_opencv_calibration_file = "zed_calibration_fixed.yml"  # Set the path to the OpenCV calibration file
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    #init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    # Start SVO recording in the run folder
    start_svo_recording(run_folder, timestamp)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = True
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    point_cloud_render = sl.Mat()
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_left = sl.Mat()
    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Utilities for tracks view
    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
    # Camera pose
    cam_w_pose = sl.Pose()

    while viewer.is_available() and not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            try:
                zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
                current_image = image_left_tmp.get_data()
                if pause_signal:
                    image_net = np.zeros_like(current_image)
                else:
                    image_net = current_image.copy()
            finally:
                lock.release()

            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            if not pause_signal:
                # Wait for detections
                lock.acquire()
                try:
                    # -- Ingest detections
                    zed.ingest_custom_box_objects(detections)
                finally:
                    lock.release()

                # Get objects from ZED SDK
                objects = sl.Objects()
                obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
                zed.retrieve_objects(objects, obj_runtime_param)
                current_objects = objects

                # Process detections for data storage
                if len(detections) > 0:
                    print("\nRaw detections:")
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Get the original SVO timestamp from the filename
                    svo_filename = os.path.basename(opt.svo)
                    original_svo_time = datetime.strptime(
                        svo_filename.split('_')[1] + "_" + svo_filename.split('_')[2].split('.')[0], "%d%m%Y_%H%M%S")

                    # Adjust current time to match original recording time
                    current_time_dt = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
                    adjusted_time = f"{original_svo_time.strftime('%Y-%m-%d')} {current_time_dt.strftime('%H:%M:%S')}"
                    current_time_dt = datetime.strptime(adjusted_time, "%Y-%m-%d %H:%M:%S")

                    # Get depth map for center point measurements
                    depth = sl.Mat()
                    point_cloud = sl.Mat()
                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

                    for i, obj in enumerate(objects.object_list):
                        if i >= len(detections):  # Safety check
                            break

                        # Get detection info from our custom box object
                        det = detections[i]
                        bbox = det.bounding_box_2d

                        # Calculate center in image coordinates
                        center_x = int((bbox[0][0] + bbox[2][0]) * 0.5)
                        center_y = int((bbox[0][1] + bbox[2][1]) * 0.5)
                        upper_y = center_y - 50
                        lower_y = center_y + 50

                        # Get distances for all three points
                        err, point3D = point_cloud.get_value(center_x, center_y)
                        center_distance = abs(point3D[2]) if err == sl.ERROR_CODE.SUCCESS else "N/A"

                        err, point3D = point_cloud.get_value(center_x, upper_y)
                        upper_distance = abs(point3D[2]) if err == sl.ERROR_CODE.SUCCESS else "N/A"

                        err, point3D = point_cloud.get_value(center_x, lower_y)
                        lower_distance = abs(point3D[2]) if err == sl.ERROR_CODE.SUCCESS else "N/A"

                        # Get object measurements
                        if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                            object_distance = float(obj.position[2])
                            try:
                                width = float(obj.dimensions[0])
                                height = float(obj.dimensions[1])
                                depth = float(obj.dimensions[2])
                            except (IndexError, TypeError) as e:
                                print(f"Error getting dimensions: {e}")
                                width = height = depth = "N/A"
                        else:
                            object_distance = width = height = depth = "N/A"

                        # Add data to DataFrame for each detection
                        new_row = pd.DataFrame({
                            'Class': [det.label],
                            'Name': [class_names[det.label]],
                            'Confidence': [f"{det.probability:.2%}"],
                            'Upper_Distance': [f"{upper_distance:.1f}" if isinstance(upper_distance, float) else "N/A"],
                            'Center_Distance': [
                                f"{center_distance:.1f}" if isinstance(center_distance, float) else "N/A"],
                            'Lower_Distance': [f"{lower_distance:.1f}" if isinstance(lower_distance, float) else "N/A"],
                            'Object_Distance': [str(object_distance)],
                            'Width': [str(width)],
                            'Height': [str(height)],
                            'Depth': [str(depth)],
                            'Timestamp': [current_time]
                        })

                        # Always add to all_detections_data
                        all_detections_data = pd.concat([all_detections_data, new_row], ignore_index=True)

                        # Check if current time falls within any pause period
                        is_paused = False
                        if opt.svo_playback and pause_periods:
                            is_paused = is_in_pause_period(current_time_dt, pause_periods)
                            if is_paused:
                                print(f"Detection in pause period: {adjusted_time}")
                            else:
                                print(f"Detection outside pause period: {adjusted_time}")

                        # Only add to filtered_detections_data if not in pause period
                        if not is_paused:
                            filtered_detections_data = pd.concat([filtered_detections_data, new_row], ignore_index=True)
                            print(f"Added detection to filtered data: {class_names[det.label]} at {adjusted_time}")
                        else:
                            print(f"Skipped detection in pause period: {class_names[det.label]} at {adjusted_time}")

                        # Save both DataFrames periodically
                        if len(all_detections_data) % 10 == 0:  # Save every 10 detections
                            all_detections_data.to_excel(all_detections_filename, index=False)
                            filtered_detections_data.to_excel(filtered_detections_filename, index=False)


                        # Print detection info
                        print(f"Class: {det.label}, Name: {class_names[det.label]}, Confidence: {det.probability:.2%}")
                        print(
                            f"Distances - Upper: {upper_distance}, Center: {center_distance}, Lower: {lower_distance}")
                        print(f"Object Distance: {object_distance}")
                        print(f"Dimensions (WxHxD): {width}mm x {height}mm x {depth}mm\n")

                    # Save data periodically (optional, adjust interval as needed)
                    all_detections_data.to_excel(all_detections_filename, index=False)
                    filtered_detections_data.to_excel(filtered_detections_filename, index=False)

            # -- Display
            # Retrieve display data
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            point_cloud.copy_to(point_cloud_render)
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 3D rendering
            viewer.updateData(point_cloud_render, current_objects if current_objects else sl.Objects())
            # 2D rendering
            np.copyto(image_left_ocv,
                      image_left.get_data() if not pause_signal else np.zeros_like(image_left.get_data()))
            cv_viewer.render_2D(image_left_ocv, image_scale, current_objects if current_objects else sl.Objects(),
                                obj_param.enable_tracking)
            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            # Tracking view
            track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

            cv2.imshow("ZED | 2D View and Birds View", global_image)
            # In the main loop, where the pause handling occurs
            key = cv2.waitKey(10)
            if key == 27:  # ESC key
                exit_signal = True
                # Save both DataFrames before exiting
                all_detections_data.to_excel(all_detections_filename, index=False)
                filtered_detections_data.to_excel(filtered_detections_filename, index=False)
            elif key == 32:  # Space bar
                if not pause_signal:  # If not paused, then pause
                    # Get current timestamp for both screenshot and Excel
                    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print("\n--- Detection Paused (Camera Blacked Out) ---")

                    # Debug prints before saving screenshot
                    print(f"Current objects available: {current_objects is not None}")
                    print(f"Detections available: {detections is not None}")

                    # Capture point cloud data before pausing
                    point_cloud_snapshot = sl.Mat()
                    zed.retrieve_measure(point_cloud_snapshot, sl.MEASURE.XYZRGBA)

                    # Save screenshot with the current data before blacking out the display
                    save_screenshot(current_image, timestamp, point_cloud_snapshot,
                                    object_distance, width, height, depth,
                                    upper_distance, center_distance, lower_distance, current_time,
                                    det, class_names)

                    # Now activate pause
                    pause_signal = True

                    # Add empty row to DataFrame to mark pause
                    detection_data = pd.concat([detection_data, pd.DataFrame(
                        {'Class': [''], 'Name': [''], 'Confidence': [''], 'Upper_Distance': [''], 'Center_Distance': [''], 'Lower_Distance': [''],
                            'Object_Distance': [''],  'Width': [''], 'Height': [''], 'Depth': [''], 'Timestamp': [f'---PAUSED--- (screenshot_{timestamp}.png)']})],
                        ignore_index=True)
                    # Save both DataFrames
                    all_detections_data.to_excel(all_detections_filename, index=False)
                    filtered_detections_data.to_excel(filtered_detections_filename, index=False)
                else:  # If paused, then unpause
                    pause_signal = False
                    print("\n--- Detection Resumed ---")

    print("Exiting...")  # Moved outside the while loop
    viewer.exit()
    exit_signal = True
    # Save both DataFrames before closing
    all_detections_data.to_excel(all_detections_filename, index=False)
    filtered_detections_data.to_excel(filtered_detections_filename, index=False)
    zed.disable_recording() # Stop recording
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--svo_playback', action='store_true', help='Running in SVO playback mode')
    parser.add_argument('--original_excel', type=str, help='Path to original Excel file with pause markers')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--use_fakemix', action='store_true', help='Enable FakeMix augmentation')
    parser.add_argument('--use_fusion', action='store_true', help='Enable Enhanced Fusion Module')
    parser.add_argument('--use_edge', action='store_true', help='Enable Edge Detection')
    parser.add_argument('--use_depth', action='store_true', help='Enable Depth-aware confidence adjustment')
    parser.add_argument('--use_multiview', action='store_true', help='Enable Multi-view Detection')
    parser.add_argument('--use_temporal', action='store_true', help='Enable Temporal Consistency')
    parser.add_argument('--config', type=str, help='Path to enhancement configuration file')
    opt = parser.parse_args()

    # If config file is provided, load enhancements from it
    if opt.config:
        enhancements = load_enhancement_config(opt.config)
        opt.use_fakemix = enhancements['use_fakemix']
        opt.use_fusion = enhancements['use_fusion']
        opt.use_edge = enhancements['use_edge']
        opt.use_depth = enhancements['use_depth']
        opt.use_multiview = enhancements['use_multiview']
        opt.use_temporal = enhancements['use_temporal']

    with torch.no_grad():
        main()
