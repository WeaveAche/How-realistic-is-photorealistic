import os
import pywt
import cv2
import numpy as np
from utils import getCentralPatch, extractFeaturesFromImg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
import pickle

class Train:
    def __init__(self, n, train_dir):
        self.n = n
        self.train_dir = train_dir

        self.i = 0

    def extractFeatures(self):
        pg_dataset = os.listdir(os.path.join(self.train_dir, "pg"))
        pr_dataset = os.listdir(os.path.join(self.train_dir, "pr"))

        self.nt = len(pg_dataset) + len(pr_dataset)
        self.features = np.zeros((self.nt, 72*(self.n-1)))
        self.clss = np.zeros(self.nt)
        
        for data in pg_dataset:
            img = cv2.imread(os.path.join(os.path.join(self.train_dir, "pg"), data))
            imgPatch = getCentralPatch(img, 256, 256)

            self.features[self.i] = extractFeaturesFromImg(self.n, imgPatch)
            self.clss[self.i] = 1
            
            self.i += 1

            print(f"Done {self.i}/{self.nt}")

        for data in pr_dataset:
            img = cv2.imread(os.path.join(os.path.join(self.train_dir, "pr"), data))
            imgPatch = getCentralPatch(img, 256, 256)

            self.features[self.i] = extractFeaturesFromImg(self.n, imgPatch)
            self.clss[self.i] = 2
            
            self.i += 1

            print(f"Done {self.i}/{self.nt}")

    def train(self):
        self.extractFeatures() 

        #with open("features", "wb") as f:
        #    pickle.dump(self.features, f)

        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)

        clf = LDA()
        clf.fit(self.features, self.clss)

        with open("models/model.pkl", "wb") as f:
            pickle.dump(clf, f)

        print(f"Model saved with training accuracy {clf.score(self.features, self.clss)*100}%")



trainer = Train(4, "./train/")
trainer.train()
