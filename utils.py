import pywt
from scipy.stats import kurtosis, skew
import numpy as np
import cv2

def getCentralPatch(img, width, height):
    h, w, c = img.shape

    left = (w - width)//2
    top = (h - height)//2
    
    return img[top : top + width, left : left + height, :]

dec_lo = [0.02807382, -0.060944743, -0.073386624, 0.41472545, 0.7973934, 0.4147245, -0.073386624, -0.060944743, 0.02807382]
dec_hi = pywt.qmf(dec_lo)
rec_lo = dec_lo
rec_hi = -dec_hi

filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi]
myWavelet = pywt.Wavelet(name = "myWavelet", filter_bank = filter_bank)

def qmf_filter(img):
    coeffs = pywt.dwt2(img, myWavelet, 'periodization')
    return coeffs

def extractFeaturesFromImg(n, img):
        features = np.zeros(72*(n - 1))

        H, V, D = [], [], []

        for i in range(3):
            Hi, Vi, Di = [], [], []

            A = img[:, :, i]
            for j in range(n):
                A, (h, v, d) = qmf_filter(A)

                Hi.append(h)
                Vi.append(v)
                Di.append(d)

            for j in range(n - 1):
                h_feat, v_feat, d_feat = getPrimaryStats(Hi[j]), getPrimaryStats(Vi[j]), getPrimaryStats(Di[j])

                start = i*12*(n - 1) + j*12

                features[start : start + 4] = h_feat
                features[start + 4 : start + 8] = v_feat
                features[start + 8 : start + 12] = d_feat

            H.append(Hi)
            V.append(Vi)
            D.append(Di)


        for j in range(3):
            for i in range(n - 1):
                start = 36*(n - 1) + j*12*(n - 1) + i*12
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

                stats = getSecondaryStats(Q_h, v_h)
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

                stats = getSecondaryStats(Q_v, v_v)
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

                stats = getSecondaryStats(Q_d, v_d)
                features[start + 8 : start + 12] = stats

        return features

def getSecondaryStats(Q, v):
    Q_t = np.transpose(Q)
    Q_tQ = np.dot(Q_t, Q)
    inv = np.linalg.inv(Q_tQ)

    w = np.dot(np.dot(inv, Q_t), v)
    p = np.log(np.abs(v)) - np.log(np.abs(np.dot(Q, w)))

    return getPrimaryStats(p)

def getPrimaryStats(arr):
    return np.array([arr.mean(), arr.var(), skew(arr, None), kurtosis(arr, None)])
