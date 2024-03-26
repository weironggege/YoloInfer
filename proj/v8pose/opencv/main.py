import cv2
import numpy as np
import random

class YOLOV8POSE:

    def __init__(self, modelpath, conf_thres=0.7, nms_thres=0.8):

        self.confThreshold = conf_thres
        self.nmsThreshold = nms_thres

        self.skeletons = [int(x.strip()) for x in open("pindex.txt", "r").readlines()]

        self.inpWidth = 640
        self.inpHeight = 640
        self.keyPnums = 17

        self.net = cv2.dnn.readNet(modelpath)

    def drawPred(self, image, box, conf):

        x,y,w,h = box

        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

        label = "Person:" + str(round(conf * 100)) + "%"

        cv2.putText(image, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    def drawPose(self, image, kps):

        for i in range(self.keyPnums):
            if kps[i * 3 + 2] > 0.5:
                cv2.circle(image, (int(kps[i * 3]), int(kps[i * 3 + 1])), 2, tuple(random.sample(range(255), 3)), 5)

        for j in range(0, len(self.skeletons), 2):
            if kps[(self.skeletons[j]-1) * 3 + 2] > 0.5:
                pos1 = (int(kps[(self.skeletons[j] - 1) * 3]), int(kps[(self.skeletons[j] - 1) * 3 + 1]))
                pos2 = (int(kps[(self.skeletons[j+1]-1) * 3]), int(kps[(self.skeletons[j+1]-1) * 3 + 1]))
                cv2.line(image, pos1, pos2, tuple(random.sample(range(255), 3)), thickness=2, lineType=cv2.LINE_AA)

    def resize_img(self, img):
        neww, newh, padw, padh = self.inpWidth, self.inpHeight, 0, 0

        srch, srcw = img.shape[:2]

        if srch != srcw:
            hw_scale = srch / srcw
            if hw_scale > 1.0:
                neww = int(self.inpWidth / hw_scale)
                timg = cv2.resize(img, (neww, newh), cv2.INTER_AREA)
                padw = int((self.inpWidth - neww) * 0.5)
                timg = cv2.copyMakeBorder(timg, 0, 0, padw, self.inpWidth-neww-padw, cv2.BORDER_CONSTANT, (114,114,114))
            else:
                newh = int(self.inpHeight * hw_scale)
                timg = cv2.resize(img, (neww, newh), cv2.INTER_AREA)
                padh = int((self.inpHeight - newh) * 0.5)
                timg = cv2.copyMakeBorder(timg, padh, self.inpHeight-newh-padh, 0, 0, cv2.BORDER_CONSTANT, (114,114,114))
        else:
            timg = cv2.resize(img, (neww, newh), cv2.INTER_AREA)
        return timg, neww, newh, padw, padh

    def detect(self, image):
        timg, neww, newh, padw, padh = self.resize_img(image)

        blob = cv2.dnn.blobFromImage(timg, 1.0 / 255.0, swapRB=True)

        self.net.setInput(blob)

        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        ratiow, ratioh = image.shape[1] / neww, image.shape[0] / newh
        boxes, confs, kpss = [], [], []

        for pred in np.squeeze(outs[0]).transpose():
            conf = pred[4]
            if conf > self.confThreshold:
                cx = (pred[0] - padw) * ratiow
                cy = (pred[1] - padh) * ratioh
                w = pred[2] * ratiow
                h = pred[3] * ratioh

                left = int(cx - 0.5 * w)
                top = int(cy - 0.5 * h)

                boxes.append([left, top, int(w), int(h)])
                confs.append(conf)

                kps = []

                for kpi in range(self.keyPnums):
                    kpx = (pred[kpi * 3 + 5] - padw) * ratiow
                    kpy = (pred[kpi * 3 + 6] - padh) * ratioh
                    kpv = pred[kpi * 3 + 7]

                    kps.append(kpx)
                    kps.append(kpy)
                    kps.append(kpv)
                kpss.append(kps)

        indices = cv2.dnn.NMSBoxes(boxes, confs, self.confThreshold, self.nmsThreshold)

        for idx in indices:
            self.drawPred(image, boxes[idx], confs[idx])
            self.drawPose(image, kpss[idx])

        return image

if __name__ == "__main__":

    net = YOLOV8POSE("weights/yolov8n-pose.onnx")

    srcimg = cv2.imread("imgs/ppose.jpg")

    tarimg = net.detect(srcimg)

    cv2.imwrite("imgs/ppose_opencv_py.jpg", tarimg)



