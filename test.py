import os
import sys
import cv2
import numpy as np
from utils import getCentralPatch, extractFeaturesFromImg
import pickle

if len(sys.argv) != 2:
    print("Usage test.py /path/to/img")
    exit(0)

img = cv2.imread(sys.argv[1])
imgPatch = getCentralPatch(img, 256, 256)
features = extractFeaturesFromImg(4, imgPatch)

features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

prob = model.predict_proba([features])[0]

if prob[0] > 0.5:
    print(f"The image is photographic with confidence {prob[0]*100}%")
else:
    print(f"The image is photorealistic with confidence {prob[1]*100}%")
