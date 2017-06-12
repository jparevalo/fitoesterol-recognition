import cv2
import numpy as np

tresh = 100
max_tresh = 255


# Detect edges using Threshold



img = cv2.imread('../Pictures/img1.tif',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

img2 = cimg

skin_row_index = 0
skin_pixel_index = 0
its_skin = False
for row in img:
    skin_pixel_index = 0
    for pixel in row:
        skin_pixel = img2[skin_row_index][skin_pixel_index]
        if skin_pixel[1] < 70:  # WHITE
            img2[skin_row_index][skin_pixel_index] = img[skin_row_index][skin_pixel_index]
        else:
            img2[skin_row_index][skin_pixel_index] = 255
        skin_pixel_index += 1
    skin_row_index += 1



#circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1, 20, param1=30, param2=28, minRadius = 10, maxRadius = 50 )

img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)



gray = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
