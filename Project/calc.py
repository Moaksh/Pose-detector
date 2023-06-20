from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint
from utils.plots import plot_skeleton_kpts

from typing import Generator, Callable, Tuple
from utils.datasets import letterbox
from torchvision import transforms

import pathlib
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import models.yolo


def squat_calc(pose_model: models.yolo.Model, angle_max: int = 150, angle_min: int = 30, threshold: int = 35) -> None:

    count=0
    cap= cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No Camera")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    stage = None
    counter = 0
    '''
    stage_right = None
    counter_right = 0'''
    while True:
        count+=1
        s, frame = cap.read()

        if not s:
            print("No Frame")
            break

        if count in range(1,100):
                continue

        frame = cv2.flip(frame, 1)
        pose_output=None
        frame, pose_output = process_frame_and_annotate(pose_model, frame, True)

        #upper body relative to thighs
        upper_body = calculate_angle(pose_output, *[1,2,3], True, frame)

        #knee angle
        legs = calculate_angle(pose_output, *[4,5,6], True, frame)

        msg=""

        if upper_body < threshold: #back less than 35
            if legs > angle_max: #knee greater than 150
                stage = 'up'
            if legs < angle_min and stage == 'up': #elbow less than 30
                stage = 'down'
                counter += 1

        else:
            stage = "skipped"
            cv2.putText(frame, "Change back position", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,
                        cv2.LINE_AA)



        # Annotation for Stage and Reps
        cv2.putText(frame, f"Stage: {stage}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (225, 225, 225), 3,
                    cv2.LINE_AA)
        cv2.putText(frame, f"Reps: {str(counter)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (225, 225, 225), 3, cv2.LINE_AA)



        cv2.imshow('camera', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()




def curl_calc(pose_model: models.yolo.Model, angle_max: int = 150, angle_min: int = 30, threshold: int = 35) -> None:

    count=0
    cap= cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No Camera")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    stage = None
    counter = 0
    '''
    stage_right = None
    counter_right = 0'''
    while True:
        count+=1
        s, frame = cap.read()

        if not s:
            print("No Frame")
            break

        if count in range(1,100):
                continue

        frame = cv2.flip(frame, 1)
        pose_output=None
        frame, pose_output = process_frame_and_annotate(pose_model, frame, True)

        #elbow angle
        left_elbow = calculate_angle(pose_output, *[6, 8, 10], True, frame)
        right_elbow = calculate_angle(pose_output, *[5, 7, 9], True, frame)

        # shoulder angle
        left_shoulder = calculate_angle(pose_output, *[12, 6, 8], True, frame, 2, threshold)
        right_shoulder = calculate_angle(pose_output, *[11, 5, 7], True, frame, 2, threshold)

        if left_shoulder < threshold: #shoulder less than 35
            if left_elbow > angle_max: #elbow greater than 150
                stage = 'down'
            if left_elbow < angle_min and stage == 'down': #elbow less than 30
                stage = 'up'
                counter += 1
        else:
            stage = "skippped"
            cv2.putText(frame, "Tuck in your left shoulder", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,
                        cv2.LINE_AA)

        if right_shoulder < threshold:
            if right_elbow > angle_max:
                stage = 'down'
            if right_elbow < angle_min and stage == 'down':
                stage = 'up'
                counter += 1
        else:
            stage = "skipped"
            cv2.putText(frame, "Tuck in your right shoulder", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,
                        cv2.LINE_AA)

        # Annotation for Stage and Reps
        cv2.putText(frame, f"Direction: {stage}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (225, 225, 225), 3,
                    cv2.LINE_AA)
        cv2.putText(frame, f"Reps: {str(counter)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (225, 225, 225), 3, cv2.LINE_AA)



        cv2.imshow('camera', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def shoulderPress(pose_model: models.yolo.Model, angle_max: int =135 , angle_min: int = 70, threshold: int = 85) -> None:

    count=0
    cap= cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No Camera")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    stage = None
    counter = 0
    '''
    stage_right = None
    counter_right = 0'''
    while True:
        count+=1
        s, frame = cap.read()

        if not s:
            print("No Frame")
            break

        if count in range(1,100):
                continue

        frame = cv2.flip(frame, 1)
        pose_output=None
        frame, pose_output = process_frame_and_annotate(pose_model, frame, True)

        #elbow angle
        left_elbow = calculate_angle(pose_output, *[6, 8, 10], True, frame)
        right_elbow = calculate_angle(pose_output, *[5, 7, 9], True, frame)

        # shoulder angle
        left_shoulder = calculate_angle(pose_output, *[12, 6, 8], True, frame, 2, threshold)
        right_shoulder = calculate_angle(pose_output, *[11, 5, 7], True, frame, 2, threshold)

        if left_shoulder > threshold:
            if left_elbow > angle_max:
                stage = 'down'

            if left_elbow < angle_min and stage == 'down':
                stage = 'up'
                counter += 1

        else:
            stage = "skippped"
            cv2.putText(frame, "Lift your left shoulder", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,
                        cv2.LINE_AA)

        if right_shoulder > threshold:
            if right_elbow > angle_max:
                stage = 'down'

            if right_elbow < angle_min and stage == 'down':
                stage = 'up'
                counter += 1
        else:
            stage = "skipped"
            cv2.putText(frame, "Lift your right shoulder", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,
                        cv2.LINE_AA)

        # Annotation for Stage and Reps
        cv2.putText(frame, f"Direction: {stage}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (225, 225, 225), 3,
                    cv2.LINE_AA)
        cv2.putText(frame, f"Reps: {str(counter)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (225, 225, 225), 3, cv2.LINE_AA)



        cv2.imshow('camera', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

############################################################################################################################
#GLOBAL PARAMS
POSE_IMAGE_SIZE = 256
STRIDE = 64
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.65


def pose_pre_process_frame(frame: np.ndarray) -> torch.Tensor:
    image = letterbox(frame, POSE_IMAGE_SIZE, stride=STRIDE, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)

    return image

#scaling
def post_process_pose(pose: np.ndarray, image_size: Tuple, scaled_image_size: Tuple) -> np.ndarray:
    height, width = image_size
    scaled_height, scaled_width = scaled_image_size
    vertical_factor = height / scaled_height
    horizontal_factor = width / scaled_width
    result = pose.copy()
    for i in range(17):
        result[i * 3] = horizontal_factor * result[i * 3]
        result[i * 3 + 1] = vertical_factor * result[i * 3 + 1]
    return result


def pose_annotate(image: np.ndarray, detections: np.ndarray) -> np.ndarray:
    annotated_frame = image.copy()

    for idx in range(detections.shape[0]):
        pose = detections[idx, 7:].T
        plot_skeleton_kpts(annotated_frame, pose, 3) #skeletal keypoints

    return annotated_frame


def pose_post_process_output(
        model: models.yolo,
        output: torch.tensor,
        confidence_threshold: float,
        iou_threshold: float,
        image_size: Tuple[int, int],
        scaled_image_size: Tuple[int, int]
) -> np.ndarray:
    output = non_max_suppression_kpt(
        prediction=output,
        conf_thres=confidence_threshold,
        iou_thres=iou_threshold,
        nc=model.yaml['nc'],
        nkpt=model.yaml['nkpt'],
        kpt_label=True)

    with torch.no_grad():
        output = output_to_keypoint(output)

        for idx in range(output.shape[0]):
            output[idx, 7:] = post_process_pose(
                output[idx, 7:],
                image_size=image_size,
                scaled_image_size=scaled_image_size
            )

    return output


def process_frame_and_annotate(model: models.yolo, frame: np.ndarray,
                               return_output: bool = False):
    pose_pre_processed_frame = pose_pre_process_frame(frame=frame.copy())

    image_size = frame.shape[:2]
    scaled_image_size = tuple(pose_pre_processed_frame.size())[2:]

    with torch.no_grad():
        pose_output, _ = model(pose_pre_processed_frame)
        pose_output = pose_post_process_output(
            model=model,
            output=pose_output,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            image_size=image_size,
            scaled_image_size=scaled_image_size
        )

    annotated_frame = pose_annotate(image=frame, detections=pose_output)
    if return_output:
        return annotated_frame, pose_output
    return annotated_frame

#########################################################################################################################
#main function to calculate angle between
def calculate_angle(pose_out: np.ndarray, a: int, b: int, c: int, draw: bool = False,
                    frame: np.ndarray = None, size=3, threshold=None) -> float:
    coord = []
    kpts=None
    #if pose_out.ndim > 1:
    kpts = pose_out[0, 7:].T
    no_kpt = len(kpts) // 3
    for i in range(no_kpt):
        cx_cy = kpts[3 * i], kpts[3 * i + 1]
        conf = kpts[3 * i + 2]
        coord.append([i, cx_cy, conf])

    a = np.array(coord[a][1])
    b = np.array(coord[b][1])
    c = np.array(coord[c][1])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    if angle > 180:
        angle = 360 - angle

    if draw and frame is not None:
        elbow = int(b[0]), int(b[1])

        if threshold:
            if angle > threshold:
                cv2.putText(frame, str(int(angle)), elbow, cv2.FONT_HERSHEY_SIMPLEX, size * 2, (0, 0, 225), 3,
                            cv2.LINE_AA)
            else:
                cv2.putText(frame, str(int(angle)), elbow, cv2.FONT_HERSHEY_SIMPLEX, size, (225, 225, 225), 3,
                            cv2.LINE_AA)
        else:
            cv2.putText(frame, str(int(angle)), elbow, cv2.FONT_HERSHEY_SIMPLEX, size, (225, 225, 225), 3, cv2.LINE_AA)

    return angle

