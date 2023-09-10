import cv2


#objective: The objective is design an app that allow you increase the resolution of you image even you low quality pixel image

sr = cv2.dnn_superres.DnnSuperResImpl.create()

def super_resolution(image, method, rescale):
    image = cv2.imread()

