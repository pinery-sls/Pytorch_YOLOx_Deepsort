import argparse
import torch
import cv2
import os
import time

import sys
sys.path.insert(0, './YOLOX')
from deep_sort_pytorch.deep_sort import build_tracker
from YOLOX.yolox.data.datasets.coco_classes import COCO_CLASSES
from YOLOX.detector import build_detector
from deep_sort_pytorch.utils.parser import get_config
import torch
import cv2
import imutils
from YOLOX.yolox.utils.visualize import vis_track

class_names = COCO_CLASSES

class VideoTracker(object):
    def __init__(self,cfg,args, filter_class=None):
        self.args=args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            raise UserWarning("Running in cpu mode!")
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()

        self.detector=build_detector(cfg,use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.filter_class = filter_class

    def __enter__(self):
        assert os.path.isfile(self.args.source), "Error: path error"
        self.vdo.open(self.args.source)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
    def run(self):
        idx_frame = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()

            # do detection
            im = imutils.resize(ori_im, height=500)
            info = self.detector.detect(im, visual=False)

            outputs = []
            if info['box_nums']>0:
                bbox_xywh = []
                scores = []
                #bbox_xywh = torch.zeros((info['box_nums'], 4))
                for (x1, y1, x2, y2), class_id, score  in zip(info['boxes'],info['class_ids'],info['scores']):
                    if self.filter_class and class_names[int(class_id)] not in self.filter_class:
                        continue
                    bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])
                    scores.append(score)
                bbox_xywh = torch.Tensor(bbox_xywh)

                # deepsort
                outputs = self.deepsort.update(bbox_xywh, scores, im)
                im = vis_track(im, outputs)

            end = time.time()
            cv2.putText(im, "FPS= "+str(int(1 / (end - start))), (0, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            print("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / (end - start)))

            if self.args.display:
                cv2.imshow("test", im)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
            
            if self.args.save_path:
                self.writer.write(im)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument("--config_detection", type=str, default="./YOLOX/yolox_s.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()

if __name__ == '__main__':
    args=parse_args()
    cfg=get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    with VideoTracker(cfg, args, filter_class=['truck','person','car']) as vdo_trk:
        vdo_trk.run()