import numpy as np
import onnxruntime
import cv2
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class YOLOV8SEG:

    def __init__(self, modelpath, conf_thres=0.7, nms_thres=0.8):
        self.confThreshold = conf_thres
        self.nmsThreshold = nms_thres

        self.class_names = [x.strip() for x in open("coco.names", "r").readlines()]

        self.inpWidth = 640
        self.inpHeight = 640

        self.nm = 32

        self.sess = onnxruntime.InferenceSession(modelpath, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        self.innames = [x.name for x in self.sess.get_inputs()]
        self.ounames = [x.name for x in self.sess.get_outputs()]


    def drawPred(self, image, box, classid, conf):

        x,y,w,h = box

        cv2.rectangle(image, (x,y), (x+w,y+h), (255,144,0), 2)

        label = self.class_names[classid] + ":" + str(round(conf, 2))

        cv2.putText(image, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,144), 2)

    def maskimg2pred(self, image, maskpreds, npboxes, maskout):
        num_mask, mask_height, mask_width = maskout.shape

        masks = sigmoid(maskpreds @ maskout.reshape((num_mask, -1))).reshape((-1, mask_height, mask_width))

        mask_img = image.copy()
        scale_npboxes = npboxes.copy().astype(np.float32)

        mratiow, mratioh = image.shape[1] / mask_width, image.shape[0] / mask_height

        blur_size = (int(mratiow), int(mratioh))

        scale_npboxes[:, [0,2]] /= mratiow
        scale_npboxes[:, [1,3]] /= mratioh

        for i in range(len(npboxes)):
            ox,oy,ow,oh = npboxes[i].astype(int)
            sx,sy,sw,sh = scale_npboxes[i].astype(int)

            scale_crop_mask = masks[i][sy:(sy+sh), sx:(sx+sw)]

            crop_mask = cv2.resize(scale_crop_mask, (ow,oh), cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)[..., np.newaxis]

            crop_mask_img = mask_img[oy:(oy+oh), ox:(ox+ow)]

            mask_img[oy:(oy+oh), ox:(ox+ow)] = crop_mask_img * (1 - crop_mask) + tuple(random.sample(range(255), 3)) * crop_mask
        maskalpha = 0.5
        return cv2.addWeighted(mask_img, maskalpha, image, 1-maskalpha, 0)

    def detect(self, image):

        blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (self.inpHeight, self.inpWidth), (0,0,0), True, False)

        outs = self.sess.run(self.ounames, {self.innames[0]:blob})

        ratiow, ratioh = image.shape[1] / self.inpWidth, image.shape[0] / self.inpHeight

        boxes, confs, ids, mpreds = [], [], [], []

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
                mpreds.append(list(pred[-self.nm:]))

        indices = cv2.dnn.NMSBoxes(boxes, confs, self.confThreshold, self.nmsThreshold)

        npboxes = np.array(boxes)[indices]
        maskpreds = np.array(mpreds)[indices]
        maskout = np.squeeze(outs[1])

        image = self.maskimg2pred(image, maskpreds, npboxes, maskout)

        for idx in indices:
            self.drawPred(image, boxes[idx], ids[idx], confs[idx])
        return image

if __name__ == "__main__":

    net = YOLOV8SEG("weights/yolov8n-seg.onnx")

    srcimg = cv2.imread("imgs/person.jpg")

    tarimg = net.detect(srcimg)

    cv2.imwrite("imgs/person_onnx_py.jpg", tarimg)
