import cv2
import numpy as np
import onnxruntime


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))



class YOLOV5:

    def __init__(self, modelpath, conf_thres=0.7, nms_thres=0.8):

        self.confThreshold = conf_thres
        self.nmsThreshold = nms_thres

        # self.class_names = [x.strip() for x in open("coco.names", "r").readlines()]
        self.class_names = ["fire"]
        self.anchors = {
            8: [[1.25000, 1.62500], [2.00000, 3.75000], [4.12500, 2.87500]],
            16: [[1.87500, 3.81250], [3.87500, 2.81250], [3.68750, 7.43750]],
            32: [[3.62500, 2.81250], [4.87500, 6.18750], [11.65625, 10.18750]]
            }

        
        self.inpWidth = 640
        self.inpHeight = 640

        self.sess = onnxruntime.InferenceSession(modelpath, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        self.innames = [x.name for x in self.sess.get_inputs()]
        self.ounames = [x.name for x in self.sess.get_outputs()]


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
                timg = cv2.resize(image, (neww, newh), cv2.INTER_AREA)
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
    
    def make_grid(self, nx, ny, stride, anchor):
        stride = np.array(stride)
        anchor = np.array(anchor)
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        grid = np.stack((xv, yv), -1)
        anchor_grid = (anchor * stride).reshape(1, len(anchor), 1, 1, 2)
        return grid, anchor_grid

    
    
    
    def postprocess(self, outs, padw, padh):
        z = []
        for out in outs:
            _, _, ny, nx, _ = out.shape
            stride = self.inpHeight / ny
            assert (stride == self.inpWidth / nx)
            anchor = self.anchors[stride]
            grid, anchor_grid = self.make_grid(nx, ny, stride, anchor)
            
            y = _sigmoid(out)
            y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + grid) * stride  # xy
            y[..., 0:1] = y[..., 0:1] - padw  # x
            y[..., 1:2] = y[..., 1:2] - padh  # y
            y[..., 2:4] = (y[..., 2:4] * 2)**2 * anchor_grid  # wh
            z.append(y.reshape(-1, 6))
        preds = np.concatenate(z, axis=0)
        return preds
            
            
    def detect(self, image):

        timg, neww, newh, padw, padh = self.resize_img(image)

        blob = cv2.dnn.blobFromImage(timg, 1.0 / 255.0, swapRB=True)
        
        outs = self.sess.run(self.ounames, {self.innames[0]:blob}) 
        
        preds = self.postprocess(outs, padw, padh)

        ratioh, ratiow = image.shape[0] / newh, image.shape[1] / neww
        boxes, ids, confs = [], [], []

        for pred in preds:
            
            if pred[4] > self.confThreshold:
                maxss = np.max(pred[5:])
                idp = np.argmax(pred[5:])
                maxss *= pred[4]
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

        indices = cv2.dnn.NMSBoxes(boxes, confs, self.confThreshold, self.nmsThreshold)

        for idx in indices:
            self.drawPred(image, boxes[idx], ids[idx], confs[idx])
        return image

if __name__ == "__main__":

    net = YOLOV5("models/lxfire_base.onnx")

    srcimg = cv2.imread("imgs/00007.jpg")

    tarimg = net.detect(srcimg)

    cv2.imwrite("imgs/00007_out3_onnx_py.jpg", tarimg)
