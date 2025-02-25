import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('/home/frederik/Videos/untitled.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def center_of_mass(binary_array):
    binary_array = np.array(binary_array)
    y_indices, x_indices = np.nonzero(binary_array)
    center_of_mass_x = np.mean(x_indices)
    center_of_mass_y = np.mean(y_indices)
    return [center_of_mass_y, center_of_mass_x]

cv2.namedWindow('1',cv2.WINDOW_NORMAL)
cv2.setWindowProperty('1',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)


A = []

while True:
    err,color_img = cap.read()
    try:
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray_img)
        ret, thresh = cv2.threshold(inv, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow("1", thresh)
        A.append(center_of_mass(thresh))
        cv2.waitKey(1)
    except:
        break
   
A = np.array(A)
plt.plot(range(len(A)), A[:,0])
plt.show()
plt.plot(range(len(np.diff(A[:,0]))), np.diff(A[:,0]))
plt.show()
