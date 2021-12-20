# import cv2

# cap = cv2.VideoCapture('sample.mp4')
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
# print(frame_width, frame_height)
# out = cv2.VideoWriter('cut.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
# print(fps)
# count = 0
# while True:
#     ret, frame = cap.read()
#     if count % 10:
#         if ret:
#             out.write(frame)
#         else:
#             break
#     count += 1
# cap.release()
# out.release()
from PIL import Image
import numpy as np
import cv2

path = 'chicago.jpg'
img = Image.open(path)
long = max(img.size)
print(img.size)
scale = 512/long
print(scale)
img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
print(img.size)

k = cv2.imread('chicago.jpg')
print(k.shape)
