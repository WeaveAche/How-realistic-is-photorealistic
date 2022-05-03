import cv2, os
import random

pg_imgs = os.listdir("./train/pg/")

total = len(pg_imgs)
req = total//5

for i in range(req):
    pg_imgs = os.listdir("./train/pg/")

    img = random.choice(pg_imgs)
    os.rename(os.path.join("./train/pg/",  img), os.path.join("./val/pg/", img))

    print(f"{i}/{req}")
