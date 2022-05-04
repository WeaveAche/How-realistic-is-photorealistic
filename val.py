import os
import pywt
import cv2
import numpy as np
from utils import getCentralPatch, extractFeaturesFromImg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
import pickle

class Val:
    def __init__(self, n, val_dir):
        self.n = n
        self.val_dir = val_dir 

        self.i = 0

    def extractFeatures(self):
        pg_dataset = os.listdir(os.path.join(self.val_dir, "pg"))
        pr_dataset = os.listdir(os.path.join(self.val_dir, "pr"))

        self.pg = len(pg_dataset)
        self.pr = len(pr_dataset)
        self.nt = self.pg + self.pr 

        self.features = np.zeros((self.nt, 72*(self.n-1)))
        self.clss = np.zeros(self.nt)
        
        for data in pg_dataset:
            img = cv2.imread(os.path.join(os.path.join(self.val_dir, "pg"), data))
            imgPatch = getCentralPatch(img, 256, 256)

            self.features[self.i] = extractFeaturesFromImg(self.n, imgPatch)
            self.clss[self.i] = 1
            
            self.i += 1

            print(f"Done {self.i}/{self.nt}")

        for data in pr_dataset:
            img = cv2.imread(os.path.join(os.path.join(self.val_dir, "pr"), data))
            imgPatch = getCentralPatch(img, 256, 256)

            self.features[self.i] = extractFeaturesFromImg(self.n, imgPatch)
            self.clss[self.i] = 2
            
            self.i += 1

            print(f"Done {self.i}/{self.nt}")

    def getAccuracy(self):
        self.extractFeatures() 

        with open("train_features", "wb") as f:
            pickle.dump(self.features, f)

        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)

        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)

        pg_accuracy = model.score(self.features[: self.pg], self.clss[: self.pg])
        pr_accuracy = model.score(self.features[self.pg :], self.clss[self.pg :])
        total_accuracy = model.score(self.features, self.clss)

        print(f"Model achieved photographic accuracy of {pg_accuracy*100}%")
        print(f"Model achieved photorealistic accuracy of {pr_accuracy*100}%")
        print(f"Total accuracy {total_accuracy*100}%")
        


val = Val(4, "./dataset/val/")
val.getAccuracy()
