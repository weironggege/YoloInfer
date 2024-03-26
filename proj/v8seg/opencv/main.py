import cv2
import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class YOLOV8SEG:

    def __init__(self, modelpath, conf_thres=0.7, nms_thres=0.8):
        self.confThreshold = conf_thres
        self.nmsThreshold = nms_thres

        self.class_names = [x.strip() for x in open("coco.names", "r").readlines()] 
        self.net = cv2.dnn.readNet(modelpath)
        self.inpHeight = 640
        self.inpWidth = 640
        
        self.nm = 32

    def drawPred(self, image, box, clsid, conf):
        x,y,w,h = box

        cv2.rectangle(image, (x,y), (x+w,y+h), (144,125,0), 2)

        label = self.class_names[clsid] + ":" + str(round(conf, 2))

        cv2.putText(image, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (90,255,67), 2)

    def detect(self, image):

        blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (self.inpHeight, self.inpWidth), (0,0,0), True, False)

        self.net.setInput(blob)

        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        ratiow, ratioh = image.shape[1] / self.inpWidth, image.shape[0] / self.inpHeight
        
        boxes, confs, ids, m_preds = [], [], [], []
        for pred in np.squeeze(outs[0]).transpose():
            maxss = np.max(pred[4:-self.nm])
            idp = np.argmax(pred[4:-self.nm])
            if maxss >= self.confThreshold:
                cx = pred[0] * ratiow
                cy = pred[1] * ratioh
                w = pred[2] * ratiow
                h = pred[3] * ratioh

                left = int(cx - 0.5 * w)
                top = int(cy - 0.5 * h)

                boxes.append([left, top, int(w), int(h)])
                confs.append(maxss)
                ids.append(idp)
                m_preds.append(list(pred[-self.nm:]))

        indices = cv2.dnn.NMSBoxes(boxes, confs, self.confThreshold, self.nmsThreshold)

        np_boxes = np.array(boxes)[indices]
        mask_preds = np.array(m_preds)[indices]
        mask_output = np.squeeze(outs[1])

        image = self.maskimg2pred(mask_preds, np_boxes, mask_output, image)

        for idx in indices:
            self.drawPred(image, boxes[idx], ids[idx], confs[idx])
        return image

    def maskimg2pred(self, mask_preds, np_boxes, mask_out, image):

        nm_mask, mask_height, mask_width = mask_out.shape

        masks = sigmoid(mask_preds @ mask_out.reshape((nm_mask, -1))).reshape((-1, mask_height, mask_width))

        mbratioh, mbratiow = image.shape[0] / mask_height, image.shape[1] / mask_width

        mask_img = image.copy()
        scale_boxes = np_boxes.copy().astype(np.float32)

        scale_boxes[:, [0,2]] /= mbratiow
        scale_boxes[:, [1,3]] /= mbratioh

        blur_size = (int(mbratiow), int(mbratioh))

        for i in range(len(np_boxes)):
            sx, sy, sw, sh = scale_boxes[i].astype(int)
            ox, oy, ow, oh = np_boxes[i].astype(int)

            scale_crop_mask = masks[i][sy:(sy+sh), sx:(sx+sw)]
            crop_mask = cv2.resize(scale_crop_mask, (ow,oh), cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)
            crop_mask = (crop_mask > 0.5).astype(np.uint8)[..., np.newaxis]

            crop_mask_img = mask_img[oy:(oy+oh), ox:(ox+ow)]

            mask_img[oy:(oy+oh), ox:(ox+ow)] = crop_mask_img * (1 - crop_mask) + tuple(random.sample(range(255), 3)) * crop_mask
        mask_alpha = 0.5
        return cv2.addWeighted(mask_img, mask_alpha, image, 1-mask_alpha, 0)


if __name__ == "__main__":


    net = YOLOV8SEG("weights/yolov8n-seg.onnx")

    srcimg = cv2.imread("imgs/person.jpg")

    tarimg = net.detect(srcimg)

    cv2.imwrite("imgs/person_opencv_py.jpg", tarimg)



