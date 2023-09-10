import glob
import os
import cv2
import numpy as np

# objective: The objective is design an app that allow you increase the resolution of you image even you low quality pixel image

# read image
org_img = cv2.imread('me1.jpg')
# Create super-resolution object.
sr = cv2.dnn_superres.DnnSuperResImpl.create()

# def super_resolution(image, method, rescale):
model_paths = glob.glob('models/*')
model_names = [os.path.basename(name).split()[0] for name in model_paths]
methods = ['edsr', 'espcn', 'fsrcnn', 'lapsrn']
model_dict = dict([(mod_name, [mod_path, mod_method]) for mod_name, mod_path, mod_method in zip(model_names, model_paths, methods)])
model_path = model_dict[model_names[1]][0]
print(model_path)
# Path to the model
sr.readModel(model_path)
# Specify the method and scale.
sr.setModel(model_dict[model_names[1]][1], 4)
# Perform Upsampling
sr_img = sr.upsample(org_img)
cv2.imwrite(f"{model_dict[model_names[1]][1]}_Upsample.jpg", sr_img)
print('Image Saved')
#
# cv2.imshow("Horizontal ", sr_img)
# # cv2.waitKey(0)
# cv2.destroyAllWindows()
