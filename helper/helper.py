from dotenv import load_dotenv
from datetime import datetime
import cv2
import base64
import io
import os



class helper:
    def __init__(self):
        load_dotenv()

        self.thresh = float(os.getenv("THRESH"))
        self.nmsThresh = float(os.getenv("NMS_THRESH"))
        self.peopleModel = str(os.getenv("PEOPLE_MODEL"))
        self.peopleModelCfg = str(os.getenv("PEOPLE_MODEL_CFG"))
        self.modelClass = str(os.getenv("MODEL_CLASS"))
        self.nnImageSize = int(os.getenv("NN_IMAGE_SIZE"))
        self.scalePercent = int(os.getenv("SCALE_PERCENT"))
        self.classes = [c.strip() for c in open(self.modelClass).readlines()]
        self.color = (255, 255,255)
        self.thickness = int(os.getenv("THICKNESS"))



    def loadModel(self):
        self.net = cv2.dnn.readNet(self.peopleModel, self.peopleModelCfg)

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(
            size = (self.nnImageSize, self.nnImageSize),
            scale = 1/255
        )
        return self.model


    def timestampStr(self):
        datetimeOBJ = datetime.now()
        timestamp = datetimeOBJ.strftime("%d-%m-%YT%H:%M:%S")
        return timestamp


    def imageReducers(self, image):
        width = int(image.shape[1] * self.scalePercent / 100)
        height = int(image.shape[0] * self.scalePercent / 100)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        return resized


    def imageConverter(self, status, image):
        '''
        0 = imageToBase64
        1 = imageToBinary
        '''
        _, jpgFrame = cv2.imencode(
            '.jpg', image,
            (
                cv2.IMWRITE_JPEG_QUALITY, 70,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            )
        )

        if status == 0:
            jpgFrameB64 = base64.b64encode(jpgFrame).decode('utf-8')
            return jpgFrameB64
        elif status == 1:
            return jpgFrame.tobytes()


    def predictToModel(self, model, image, filter):
        counter = 0
        confidence = []

        img = cv2.imread(image)     

        classes, scores, boxes = model.detect(
            img, self.thresh, self.nmsThresh
        )

        for (classid, score, box) in zip(classes, scores, boxes):

            if self.classes[classid[0]] == filter:
                cv2.rectangle(img, box, self.color, self.thickness)

                confidence.append(score[0]*100)
                counter +=1
        
        if counter == 0:
            return 'No Detection', 0, 0

        return img, counter, round(
            (sum(confidence)/len(confidence)), 2
        )


    
