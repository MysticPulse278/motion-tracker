from openpiv import piv, tools
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2


frame_a  = tools.imread('cap_untitled_00:00:02_01.png' )
print(frame_a)
plt.imshow(frame_a)
plt.show()

frame_b  = tools.imread('cap_untitled_00:00:02_02.png')

#x, frame_a = cv2.threshold(frame_a, 150, 255, cv2.THRESH_BINARY)
plt.imshow(frame_a)
plt.show()
#x, frame_b = cv2.threshold(frame_b, 150, 255, cv2.THRESH_BINARY)


piv.simple_piv(frame_a, frame_b)

