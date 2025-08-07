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


# Create DataFrame for storing detections
detection_data = pd.DataFrame(columns=['Class', 'Name', 'Confidence', 'Distance', 'Timestamp'])

def save_to_excel():
    global detection_data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detections_{timestamp}.xlsx"
    detection_data.to_excel(filename, index=False)
    print(f"\nData saved to {filename}")


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


def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections, class_names, zed, detection_data, pause_signal

    print("Intializing Network...")

    model = YOLO(weights)
    class_names = model.names  # Store class names in global variable
    print("Class mapping:", class_names)  # Print class mapping at startup

    while not exit_signal:
        if run_signal and not pause_signal:
            lock.acquire()

            img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)
            # https://docs.ultralytics.com/modes/predict/#video-suffixes
            det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes

            # Debug print to see detections
            print("\nRaw detections:")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for d in det:
                class_id = int(d.cls.item())  # Extract scalar value from numpy array
                conf = float(d.conf.item())  # Extract confidence as float

                # Get center point of detection for depth measurement
                xywh = d.xywh[0]
                center_x = int(xywh[0])
                center_y = int(xywh[1])

                # Get depth at the center point of detection
                depth = sl.Mat()
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                distance = depth.get_value(center_x, center_y)[1]  # Get depth value at center point

                print(f"Class: {class_id}, Name: {model.names[class_id]}, Confidence: {conf:.2%}, Distance: {distance:.4f}m")

                # Add detection to DataFrame
                new_row = pd.DataFrame({
                    'Class': [class_id],
                    'Name': [model.names[class_id]],
                    'Confidence': [f"{conf:.2%}"],
                    'Distance': [f"{distance:.4f}"],
                    'Timestamp': [current_time]
                })
                detection_data = pd.concat([detection_data, new_row], ignore_index=True)

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, image_net)
            lock.release()
            run_signal = False
        sleep(0.01)


def main():
    global image_net, exit_signal, run_signal, detections, class_names, zed, pause_signal, detection_data

    zed = sl.Camera()  # Initialize ZED camera

    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    print("Initializing Camera...")
    #zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    #init_params.optional_opencv_calibration_file = "zed_calibration_fixed.yml"  # Set the path to the OpenCV calibration file
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    #init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

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
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            if not pause_signal:  # Only ingest detections if not paused
                zed.ingest_custom_box_objects(detections)
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)

            # -- Display
            # Retrieve display data
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            point_cloud.copy_to(point_cloud_render)
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 3D rendering
            viewer.updateData(point_cloud_render, objects)
            # 2D rendering
            np.copyto(image_left_ocv, image_left.get_data())
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            # Tracking view
            track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

            cv2.imshow("ZED | 2D View and Birds View", global_image)
            key = cv2.waitKey(10)
            if key == 27:  # ESC key
                exit_signal = True
                save_to_excel()  # Save data when exiting
            elif key == 32:  # Space bar
                pause_signal = not pause_signal
                if pause_signal:
                    print("\n--- Detection Paused ---")
                    # Add empty row to DataFrame to mark pause
                    detection_data = pd.concat([detection_data, pd.DataFrame(
                        {'Class': [''], 'Name': [''], 'Confidence': [''], 'Distance': [''],
                         'Timestamp': ['---PAUSED---']})], ignore_index=True)
                else:
                    print("\n--- Detection Resumed ---")
            else:
                # Only exit if there's an error with grab
                if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                    exit_signal = True
                    save_to_excel()  # Save data when exiting

    viewer.exit()
    exit_signal = True
    save_to_excel()  # Final save before closing
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
