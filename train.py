import os
import pywt
import cv2
import numpy as np
from scipy.stats import kurtosis, skew
from utils import qmf_filter, getCentralPatch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
import pickle

class Train:
    def __init__(self, n, train_dir):
        self.n = n
        self.train_dir = train_dir

        self.i = 0

    def extractFeaturesFromImg(self, img, debug = False): 
        if debug:
            print(f"img shape = {img.shape}")

        features = np.zeros(72*(self.n - 1))

        H, V, D = [], [], []
        
        for i in range(3):
            Hi, Vi, Di = [], [], []

            A = img[:, :, i]
            for j in range(self.n):
                A, (h, v, d) = qmf_filter(A)

                if debug:
                    print(f"h,v,d = {h.shape}, {v.shape}, {d.shape}")

                Hi.append(h)
                Vi.append(v)
                Di.append(d)

            for j in range(self.n - 1):
                h_feat, v_feat, d_feat = self.getPrimaryStats(Hi[j]), self.getPrimaryStats(Vi[j]), self.getPrimaryStats(Di[j])

                start = i*12*(self.n - 1) + j*12

                features[start : start + 4] = h_feat
                features[start + 4 : start + 8] = v_feat
                features[start + 8 : start + 12] = d_feat
            
            H.append(Hi)
            V.append(Vi)
            D.append(Di)


        for j in range(3): 
            for i in range(self.n - 1):
                start = 36*(self.n - 1) + j*12*(self.n - 1) + i*12
                dim = V[j][i].shape[0]

                Q_h = np.zeros((dim*dim, 9))
                v_h = H[j][i].reshape((dim*dim, 1))

                for k in range(dim*dim):
                    x = k//dim
                    y = k%dim

                    row = np.array([
                                    abs(0 if x == 0 else H[j][i][x-1][y]), 
                                    abs(0 if x == dim -1 else H[j][i][x+1][y]), 
                                    abs(0 if y == 0 else H[j][i][x][y-1]), 
                                    abs(0 if y == dim -1 else H[j][i][x][y+1]), 
                                    abs(H[j][i+1][x//2][y//2]), 
                                    abs(D[j][i][x][y]), 
                                    abs(D[j][i+1][x//2][y//2]), 
                                    abs(H[(j+1)%3][i][x][y]), 
                                    abs(H[(j+2)%3][i][x][y])
                                    ])

                    Q_h[k, :] = row

                stats = self.getSecondaryStats(Q_h, v_h)
                features[start : start + 4] = stats

                Q_v = np.zeros((dim*dim, 9))
                v_v = V[j][i].reshape((dim*dim, 1))

                for k in range(dim*dim):
                    x = k//dim
                    y = k%dim

                    row = np.array([
                                    abs(0 if x == 0 else V[j][i][x-1][y]), 
                                    abs(0 if x == dim -1 else V[j][i][x+1][y]), 
                                    abs(0 if y == 0 else V[j][i][x][y-1]), 
                                    abs(0 if y == dim -1 else V[j][i][x][y+1]), 
                                    abs(V[j][i+1][x//2][y//2]), 
                                    abs(D[j][i][x][y]), 
                                    abs(D[j][i+1][x//2][y//2]), 
                                    abs(V[(j+1)%3][i][x][y]), 
                                    abs(V[(j+2)%3][i][x][y])
                                    ])

                    Q_v[k, :] = row

                stats = self.getSecondaryStats(Q_v, v_v)
                features[start + 4 : start + 8] = stats

                Q_d = np.zeros((dim*dim, 9))
                v_d = D[j][i].reshape((dim*dim, 1))

                for k in range(dim*dim):
                    x = k//dim
                    y = k%dim

                    row = np.array([
                                    abs(0 if x == 0 else D[j][i][x-1][y]), 
                                    abs(0 if x == dim - 1 else D[j][i][x+1][y]), 
                                    abs(0 if y == 0 else D[j][i][x][y-1]), 
                                    abs(0 if y == dim - 1 else D[j][i][x][y+1]), 
                                    abs(D[j][i+1][x//2][y//2]), 
                                    abs(H[j][i][x][y]), 
                                    abs(V[j][i][x][y]), 
                                    abs(D[(j+1)%3][i][x][y]), 
                                    abs(D[(j+2)%3][i][x][y])
                                    ])

                    Q_d[k, :] = row

                stats = self.getSecondaryStats(Q_d, v_d)
                features[start + 8 : start + 12] = stats

        self.features[self.i, :] = features
    
    def getSecondaryStats(self, Q, v):
        Q_t = np.transpose(Q)
        Q_tQ = np.dot(Q_t, Q)
        inv = np.linalg.inv(Q_tQ)

        w = np.dot(np.dot(inv, Q_t), v) 
        p = np.log(np.abs(v)) - np.log(np.abs(np.dot(Q, w)))

        return self.getPrimaryStats(p)

    def getPrimaryStats(self, arr):
        return np.array([arr.mean(), arr.var(), skew(arr, None), kurtosis(arr, None)])

    def extractFeatures(self):
        pg_dataset = os.listdir(os.path.join(self.train_dir, "pg"))
        pr_dataset = os.listdir(os.path.join(self.train_dir, "pr"))

        self.nt = len(pg_dataset) + len(pr_dataset)
        self.features = np.zeros((self.nt, 72*(self.n-1)))
        self.clss = np.zeros(self.nt)
        
        for data in pg_dataset:
            img = cv2.imread(os.path.join(os.path.join(self.train_dir, "pg"), data))
            imgPatch = getCentralPatch(img, 256, 256)

            self.extractFeaturesFromImg(imgPatch)

            self.clss[self.i] = 1
            
            self.i += 1

            print(f"Done {self.i}/{self.nt}")

        for data in pr_dataset:
            img = cv2.imread(os.path.join(os.path.join(self.train_dir, "pr"), data))
            imgPatch = getCentralPatch(img, 256, 256)

            self.extractFeaturesFromImg(imgPatch)
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
