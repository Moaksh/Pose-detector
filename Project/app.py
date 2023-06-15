import cv2
import time
import torch
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer
from torchvision import transforms
import cv2
import math



def find_angle(imag, kpts, p1, p2, p3, draw = False):
    coord = []
    no_kpt = len(kpts)//3
    for i in range(no_kpt):
        cx, cy = kpts[i*3], kpts[i*3+1]
        conf = kpts[i*3+2]
        coord.append([cx, cy, conf])

    points = (p1,p2,p3)
    x1, y1 = coord[p1][1:3]
    x2, y2 = coord[p2][1:3]
    x3, y3 = coord[p3][1:3]

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360

    if draw:
        cv2.line(imag, (int(x1), int(y1)), (int(x2), int(y2)), (254,118,136), 3)
        cv2.line(imag, (int(x3), int(y3)), (int(x2), int(y2)), (254,118,136), 3)

        cv2.circle(imag, (int(x1), int(y1)), 10, (254,118,136), cv2.FILLED)
        cv2.circle(imag, (int(x1), int(y1)), 20, (254,118,136), 5)
        cv2.circle(imag, (int(x2), int(y2)), 10,(254,118,136), cv2.FILLED)
        cv2.circle(imag, (int(x2), int(y2)), 20, (254,118,136), 5)
        cv2.circle(imag, (int(x3), int(y3)), 10, (254,118,136), cv2.FILLED)
        cv2.circle(imag, (int(x3), int(y3)), 20, (254,118,136), 5)

    return int(angle)


@torch.no_grad()
def run(poseweights='yolov7-w6-pose.pt', source='pose.mp4', device='cpu',curltracker=True, findangle=False):
    path = source
    ext = path.split('/')[-1].split('.')[-1].strip().lower()
    if ext in ["mp4", "webm", "avi"] or ext not in ["mp4", "webm", "avi"] and ext.isnumeric():
        input_path = int(path) if path.isnumeric() else path
        device = select_device(device)
        model = attempt_load(poseweights, map_location=device)
        _ = model.eval()

        cap = cv2.VideoCapture(input_path)

        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

        vid_write_image = letterbox(
            cap.read()[1], (frame_width), stride=64, auto=True)[0]
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = "output" if path.isnumeric else f"{input_path.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{out_video_name}_result4.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30,
                              (resize_width, resize_height))

        frame_count, total_fps = 0, 0

        bcount = 0
        direction = 0



        while cap.isOpened:
            ret, frame = cap.read()
            if ret:
                orig_image = frame

                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))

                image = image.to(device)
                image = image.float()
                start_time = time.time()

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'],
                                                 kpt_label=True)
                output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # ========PushUps==========
                if curltracker:
                    for idx in range(output.shape[0]):
                        kpts = output[idx, 7:].T
                        angle = find_angle(img, kpts, 5, 7, 9)
                        percentage = np.interp(angle, (210, 290), (0, 100))
                        bar = np.interp(angle, (220, 290), (int(frame_height)-100, 100))

                        # plot_skeleton_kpts(img,kpts,3) #draws skeleton around the body


                        #check is pushup is done
                        if percentage == 100:
                            if direction == 0:
                                bcount += 0.5
                                direction = 1
                        if percentage == 0:
                            if direction == 1:
                                bcount += 0.5
                                direction = 0

                        # color = (254,118,136)
                        # font = "arial.ttf"
                        #
                        # cv2.line(img, (100, 100), (100, int(frame_height) - 100), (255, 255, 255), 30)
                        # cv2.line(img, (100, int(bar)), (100, int(frame_height) - 100), color, 30)

                        # draw.text((145, int(bar) - 17), f"{int(percentage)}%", font=font, fill=(255, 255, 255))
                        # draw.text((frame_width - 230, (frame_height // 2) - 100), f"{int(bcount)}", font=font, fill=(255, 255, 255))

                        # draw bar
                        cv2.rectangle(img, (int(frame_width)-50, 100), (int(frame_width)-20, int(frame_height)-100), (254,118,136), 30)
                        cv2.rectangle(img, (int(frame_width)-50, int(bar)), (int(frame_width)-20, int(frame_height)-100), (254,118,136), cv2.FILLED)
                        cv2.putText(img, f'{int(percentage)}%', (int(frame_width)-50, int(frame_height)-50), cv2.FONT_HERSHEY_PLAIN, 2, (254,118,136), 2)
                    if findangle:
                        print(angle, percentage,bcount, direction)
                for idx in range(output.shape[0]):
                    plot_skeleton_kpts(img, output[idx, 7:].T, 3)

                if ext.isnumeric():
                    cv2.imshow("Detection", img)
                    key = cv2.waitKey(1)
                    if key == ord('c'):
                        break

                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                out.write(img)
            else:
                break

        cap.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, help='path to video or 0 for webcam')
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')
    parser.add_argument('--findangle', type=str, default=False, help='find angle between two points')

    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
