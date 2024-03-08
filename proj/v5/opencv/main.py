import cv2
import numpy as np

class YOLOV5:

    def __init__(self, modelpath, conf_thres=0.7, nms_thres=0.8):

        self.confThreshold = conf_thres
        self.nmsThreshold = nms_thres

        self.class_names = [x.strip() for x in open("coco.names", "r").readlines()]

        self.inpWidth = 640
        self.inpHeight = 640

        self.net = cv2.dnn.readNet(modelpath)

    def drawPred(self, image, box, classid, conf):
        x,y,w,h = box

        cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 2)

        label = self.class_names[classid] + ":" + str(round(conf * 100)) + "%"

        cv2.putText(image, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    def resize_img(self, image):
        neww, newh, padw, padh = self.inpWidth, self.inpHeight, 0, 0

        srch, srcw = image.shape[:2]

        if srch != srcw:
            hw_scale = srch / srcw
            if hw_scale > 1.0:
                neww = int(self.inpWidth / hw_scale)
                timg = cv2.resize(image, (neww, newh), CV2.INTER_AREA)
                padw = int((self.inpWidth - neww) * 0.5)
                timg = cv2.copyMakeBorder(timg, 0, 0, padw, self.inpWidth-neww-padw, cv2.BORDER_CONSTANT, (114,114,114))
            else:
                newh = int(self.inpHeight * hw_scale)
                timg = cv2.resize(image, (neww, newh), cv2.INTER_AREA)
                padh = int((self.inpHeight - newh) * 0.5)
                timg = cv2.copyMakeBorder(timg, padh, self.inpHeight-newh-padh, 0, 0, cv2.BORDER_CONSTANT, (114,114,114))
        else:
            timg = cv2.resize(image, (neww, newh), cv2.INTER_AREA)

        return timg, neww, newh, padw, padh

    def detect(self, image):

        timg, neww, newh, padw, padh = self.resize_img(image)

        blob = cv2.dnn.blobFromImage(timg, 1.0 / 255.0, swapRB=True)

        self.net.setInput(blob)

        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        ratioh, ratiow = image.shape[0] / newh, image.shape[1] / neww
        boxes, ids, confs = [], [], []

        for pred in np.squeeze(outs[0]):
            if pred[4] > self.confThreshold:
                maxss = np.max(pred[5:])
                idp = np.argmax(pred[5:])
                maxss *= pred[4]
                if maxss >= self.confThreshold:
                    cx = (pred[0] - padw) * ratiow
                    cy = (pred[1] - padh) * ratioh
                    w = pred[2] * ratiow
                    h = pred[3] * ratioh

                    left = int(cx - 0.5 * w)
                    top = int(cy - 0.5 * h)

                    boxes.append([left, top, int(w), int(h)])
                    confs.append(maxss)
                    ids.append(idp)

        indices = cv2.dnn.NMSBoxes(boxes, confs, self.confThreshold, self.nmsThreshold)

        for idx in indices:
            self.drawPred(image, boxes[idx], ids[idx], confs[idx])
        return image

if __name__ == "__main__":

    net = YOLOV5("weights/yolov5stmp.onnx")

    srcimg = cv2.imread("imgs/person.jpg")

    tarimg = net.detect(srcimg)

    cv2.imwrite("imgs/person_opencv_py.jpg", tarimg)


