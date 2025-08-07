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

# Create DataFrame for storing detections
detection_data = pd.DataFrame(columns=['Class', 'Name', 'Confidence', 'Center_Distance', 'Object_Distance',
                                     'Width', 'Height', 'Depth', 'Timestamp'])

def save_to_excel():
    global detection_data, excel_filename
    if excel_filename is None:
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        excel_filename = f"detections/detections_{timestamp}.xlsx"
    detection_data.to_excel(excel_filename, index=False)
    print(f"\nData saved to {excel_filename}")


def save_screenshot(image, timestamp, point_cloud):
    """Save a screenshot with a timestamp-based filename and center point markers"""
    global current_objects, detections

    # Create a copy of the image to draw on
    display_image = image.copy()

    # Debug prints
    print(f"Number of current objects: {len(current_objects.object_list) if current_objects else 0}")
    print(f"Number of detections: {len(detections) if detections else 0}")
    print(f"Image shape: {display_image.shape}")  # Add this debug print

    # Draw center points for each detection if available
    if current_objects and detections and len(current_objects.object_list) > 0 and len(detections) > 0:
        for i, obj in enumerate(current_objects.object_list):
            if i >= len(detections):
                break

            det = detections[i]
            bbox = det.bounding_box_2d

            # Calculate center in image coordinates - FIXED calculation
            center_x = int((bbox[0][0] + bbox[2][0]) * 0.5)  # Removed multiplication by image width
            center_y = int((bbox[0][1] + bbox[2][1]) * 0.5)  # Removed multiplication by image height

            print(f"Bounding box coordinates: {bbox}")  # Debug print
            print(f"Drawing marker at ({center_x}, {center_y})")

            # Ensure coordinates are within image bounds
            height, width = display_image.shape[:2]
            center_x = max(0, min(center_x, width - 1))
            center_y = max(0, min(center_y, height - 1))

            # Draw a more visible cross marker at the center point
            color = (255, 0, 0)  # Changed to RED color (RGB format) for better visibility
            size = 20  # Increased size
            thickness = 3  # Increased thickness

            # Draw the cross
            cv2.line(display_image, (center_x - size, center_y), (center_x + size, center_y), color, thickness)
            cv2.line(display_image, (center_x, center_y - size), (center_x, center_y + size), color, thickness)

            # Draw a small circle at the center for better visibility
            cv2.circle(display_image, (center_x, center_y), 5, color, -1)

            # Add the distance measurement text near the marker
            err, point3D = point_cloud.get_value(center_x, center_y)
            if err == sl.ERROR_CODE.SUCCESS:
                distance_text = f"{abs(point3D[2]):.1f}mm"
                cv2.putText(display_image, distance_text, (center_x + 25, center_y + 5),    # Position the text to the right of the cross
                #cv2.putText(display_image, distance_text, (center_x - 40, center_y - 20),   # Position the text above the cross
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                print(f"Added distance text: {distance_text}")

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

                        # Get measurements using point cloud
                        err, point3D = point_cloud.get_value(center_x, center_y)
                        center_distance = point3D[2] if err == sl.ERROR_CODE.SUCCESS else 0.0

                        # Get object measurements
                        if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                            object_distance = float(obj.position[2])

                            # Get dimensions directly from the object
                            try:
                                width = float(obj.dimensions[0])    # x dimension in mm
                                height = float(obj.dimensions[1])   # y dimension in mm
                                depth = float(obj.dimensions[2])    # z dimension in mm
                            except (IndexError, TypeError) as e:
                                print(f"Error getting dimensions: {e}")
                                width = height = depth = "N/A"
                        else:
                            object_distance = width = height = depth = "N/A"

                            # Use the label and confidence from our custom box object
                        class_id = det.label
                        confidence = det.probability

                        print(f"Class: {class_id}, Name: {class_names[class_id]}, Confidence: {confidence:.2%}, "
                              f"Center Distance: {center_distance:.4f}mm, Object Distance: {object_distance}mm\n"
                              f"Dimensions (WxHxD): {width}mm x {height}mm x {depth}mm")

                        # Add detection to DataFrame
                        new_row = pd.DataFrame({
                            'Class': [class_id],
                            'Name': [class_names[class_id]],
                            'Confidence': [f"{confidence:.2%}"],
                            'Center_Distance': [f"{center_distance:.4f}"],
                            'Object_Distance': [str(object_distance)],
                            'Width': [str(width)],
                            'Height': [str(height)],
                            'Depth': [str(depth)],
                            'Timestamp': [current_time]
                        })
                        detection_data = pd.concat([detection_data, new_row], ignore_index=True)

                    # Update the objects for display
                    current_objects = objects

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
                    print("\n--- Detection Paused (Camera Blacked Out) ---")

                    # Debug prints before saving screenshot
                    print(f"Current objects available: {current_objects is not None}")
                    print(f"Detections available: {detections is not None}")

                    # Capture point cloud data before pausing
                    point_cloud_snapshot = sl.Mat()
                    zed.retrieve_measure(point_cloud_snapshot, sl.MEASURE.XYZRGBA)

                    # Save screenshot with the current data before blacking out the display
                    save_screenshot(current_image, timestamp, point_cloud_snapshot)

                    # Now activate pause
                    pause_signal = True

                    # Add empty row to DataFrame to mark pause
                    detection_data = pd.concat([detection_data, pd.DataFrame(
                        {'Class': [''], 'Name': [''], 'Confidence': [''],
                            'Center_Distance': [''], 'Object_Distance': [''], 'Timestamp': [f'---PAUSED--- (screenshot_{timestamp}.png)']})],
                        ignore_index=True)
                    save_to_excel()  # Save data when pausing
                else:  # If paused, then unpause
                    pause_signal = False
                    print("\n--- Detection Resumed ---")

    print("Exiting...")  # Moved outside the while loop
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
