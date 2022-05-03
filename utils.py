import pywt
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

