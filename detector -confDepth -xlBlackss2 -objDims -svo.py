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

# Update DataFrame columns at the start of the program
detection_data = pd.DataFrame(columns=['Class', 'Name', 'Confidence',
                                     'Upper_Distance', 'Center_Distance', 'Lower_Distance',
                                     'Object_Distance', 'Width', 'Height', 'Depth', 'Timestamp'])

def save_to_excel():
    global detection_data, excel_filename
    if excel_filename is None:
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        excel_filename = f"detections/detections_{timestamp}.xlsx"
    detection_data.to_excel(excel_filename, index=False)
    print(f"\nData saved to {excel_filename}")


def start_svo_recording():
    global svo_recording, excel_filename
    if excel_filename is None:
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        excel_filename = f"detections/detections_{timestamp}.xlsx"

    # Extract timestamp from excel filename
    timestamp = excel_filename.split('_')[1] + "_" + excel_filename.split('_')[2].split('.')[0]
    svo_path = f"detections/recording_{timestamp}.svo"

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

    filename = f"detections/screenshot_{timestamp}.png"
    cv2.imwrite(filename, cv2.cvtColor(display_image, cv2.COLOR_RGBA2BGR))
    print(f"Screenshot saved as {filename}")


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

# Add to global variables at top
current_objects = None

def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections, class_names, zed, detection_data, pause_signal, current_objects

    print("Intializing Network...")

    model = YOLO(weights)
    class_names = model.names
    print("Class mapping:", class_names)

    while not exit_signal:
        if run_signal:
            lock.acquire()
            try:
                img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)
                det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes

                if not pause_signal and len(det) > 0:
                    # Create custom box objects
                    detections = detections_to_custom_box(det, image_net)
                else:
                    detections = []
            finally:
                lock.release()
            run_signal = False
        sleep(0.01)


def main():
    global image_net, exit_signal, run_signal, detections, class_names, zed, pause_signal, detection_data, excel_filename, current_objects

    save_to_excel()  # Create initial Excel file

    zed = sl.Camera()  # Initialize ZED camera

    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
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

    start_svo_recording() # Start SVO recording

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
                        detection_data = pd.concat([detection_data, new_row], ignore_index=True)

                        # Print detection info
                        print(f"Class: {det.label}, Name: {class_names[det.label]}, Confidence: {det.probability:.2%}")
                        print(
                            f"Distances - Upper: {upper_distance}, Center: {center_distance}, Lower: {lower_distance}")
                        print(f"Object Distance: {object_distance}")
                        print(f"Dimensions (WxHxD): {width}mm x {height}mm x {depth}mm\n")

                    # Save data periodically (optional, adjust interval as needed)
                    save_to_excel()

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
                save_to_excel()  # Save data when exiting
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
                    save_to_excel()  # Save data when pausing
                else:  # If paused, then unpause
                    pause_signal = False
                    print("\n--- Detection Resumed ---")

    print("Exiting...")  # Moved outside the while loop
    viewer.exit()
    exit_signal = True
    save_to_excel()  # Final save before closing
    zed.disable_recording() # Stop recording
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')

    opt = parser.parse_args()

    with torch.no_grad():
        main()
