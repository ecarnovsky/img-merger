import cv2
import numpy as np 

top_img = cv2.imread("images/eagle.jpg")
bottom_img = cv2.imread("images/forrest.jpg")
# Create an all white mask
mask = 255 * np.ones(top_img.shape, top_img.dtype)

print("The dimensions of the top image are: " , top_img.shape) 
print("The dimensions of the bottom image are: " , bottom_img.shape)
 

# width, height, channels = bottom_img.shape
# center = (width // 2, height // 2)

center_of_top_img = (800, 200) 

print("The center point of the bottom image is: " , center_of_top_img)
 
# Seamlessly clone src into dst and put the results in output
normal_clone = cv2.seamlessClone(top_img, bottom_img, mask, center_of_top_img, cv2.NORMAL_CLONE)
mixed_clone = cv2.seamlessClone(top_img, bottom_img, mask, center_of_top_img, cv2.MIXED_CLONE)
 
# Write results
cv2.imwrite("images/result-normal-clone.jpg", normal_clone)
cv2.imwrite("images/result-mixed-clone.jpg", mixed_clone)