import cv2
import numpy as np


image = cv2.imread('0ad1b5b5b16d7cc869a10833fb5f700e.jpg')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


blurred = cv2.GaussianBlur(gray, (5, 5), 0)


edges = cv2.Canny(blurred, 50, 150)

result  = []
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)


for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    result.append(line)

if len(result) <0:
    print("Fotoğraf kınalı el.")
else:
    print("Fotoğraf kınasız el.")

cv2.imshow('El Çizgileri', image)
cv2.waitKey(0)
cv2.destroyAllWindows()