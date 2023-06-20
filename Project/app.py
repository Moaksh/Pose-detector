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



def find_angle(imag, kpts, p1, p2, p3):
    coord = []
    no_kpt = len(kpts)//3
    for i in range(no_kpt):
        cx, cy = kpts[i*3], kpts[i*3+1]
        conf = kpts[i*3+2]
        coord.append([cx, cy, conf])

    # points = (p1,p2,p3)
    x1, y1 = coord[p1][1:3]
    x2, y2 = coord[p2][1:3]
    x3, y3 = coord[p3][1:3]

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360

    return int(angle)


@torch.no_grad()
def run(poseweights='yolov7-w6-pose.pt', source='pose.mp4', device='cpu',curltracker=False,pushuptracker = True,  findangle=False):
    path = source
    ext = path.split('/')[-1].split('.')[-1].strip().lower()
    if ext in ["mp4", "webm", "avi"] or ext not in ["mp4", "webm", "avi"] and ext.isnumeric():
        input_path = int(path) if ext.isnumeric() else path
        device = select_device(device)
        model = attempt_load(poseweights, map_location=device)
        _ = model.float().eval()

        vid = cv2.VideoCapture(input_path)

        frame_width, frame_height = int(vid.get(3)), int(vid.get(4))

        vid_write_image = letterbox(
            vid.read()[1], (frame_width), stride=64, auto=True)[0]
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = "output"
        out = cv2.VideoWriter(f"{out_video_name}_result4.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30,
                              (resize_width, resize_height))

        frame_count, total_fps = 0, 0
        push_ups = 0
        direction = 0
        bar = 0
        percentage = 0

        fontpath = "futur.ttf"
        font = ImageFont.truetype(fontpath, 32)

        font1 = ImageFont.truetype(fontpath, 160)


        while vid.isOpened:
            ret, frame = vid.read()
            if ret:
                orig_image = frame

                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                # image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))

                image = image.to(device)
                image = image.float()
                # start_time = time.time()

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'],
                                                 kpt_label=True)
                with torch.no_grad():
                    output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # ========PushUps==========
                if pushuptracker:
                    for idx in range(output.shape[0]):
                        kpts = output[idx, 7:].T
                        angleR = find_angle(img, kpts, 5, 7, 9)
                        angleL = find_angle(img, kpts, 6, 8, 10)
                        percentage = np.interp(angleR, (210, 290), (0, 100))
                        bar = np.interp(angleR, (220, 290), (int(frame_height)-100, 100))

                        # plot_skeleton_kpts(img,kpts,3) #draws skeleton around the body


                        #check is pushup is done
                        if direction == 0:
                            if percentage == 100:
                                push_ups += 0.5

                                direction = 1

                        if direction == 1:
                            if percentage == 0:
                                push_ups += 0.5

                                direction = 0

                        if findangle:
                            print(angleR, angleL, percentage)

                        cv2.line(img, (100, 100), (100, int(frame_height) - 100), (128, 128, 128), 30)
                        cv2.line(img, (100, int(bar)), (100, int(frame_height) - 100), (128,0,0), 30)

                        if (int(percentage) < 10):
                            cv2.line(img, (155, int(bar)), (190, int(bar)), (128,0,0), 40)
                        elif ((int(percentage) >= 10) and (int(percentage) < 100)):
                            cv2.line(img, (155, int(bar)), (200, int(bar)), (128,0,0), 40)
                        else:
                            cv2.line(img, (155, int(bar)), (210, int(bar)), (128,0,0), 40)


                    im = Image.fromarray(img)
                    draw = ImageDraw.Draw(im)
                    # draw.rounded_rectangle((frame_width-300,(frame_height//2)+100 , frame_width-50,(frame_height//2)+100),fill = (128,0,0),radius = 40)

                    draw.text((145, int(bar) - 17), f"{int(percentage)}%", font=font, fill=(255, 255, 255))

                    draw.text((frame_width - 300, (frame_height // 2) - 250), f"{int(push_ups)/25}", font=font1, fill=(128, 0, 0))

                    img = np.array(im)
                    # #percentage of pushups done
                    #     if bcount == 0:
                    #         cv2.putText(img, f"Pushups: {bcount}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #     else:
                    #         cv2.putText(img, f"Pushups: {bcount}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    #
                    #     #draws bar
                    #     cv2.rectangle(img, (int(frame_width)-100, 100), (int(frame_width)-50, int(bar)), (0, 255, 0), cv2.FILLED)
                    #     cv2.putText(img, f"{int(percentage)}%", (int(frame_width)-90, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)







                for idx in range(output.shape[0]):
                    plot_skeleton_kpts(img, output[idx, 7:].T, 3)



                if ext.isnumeric():
                    cv2.imshow("Detection", img)
                    key = cv2.waitKey(1)
                    if key == ord('c'):
                        break

                # end_time = time.time()
                # fps = 1 / (end_time - start_time)
                # total_fps += fps
                # frame_count += 1
                out.write(img)
            else:
                break

        vid.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt')
    parser.add_argument('--source', type=str)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--findangle', type=str, default=False)

    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
