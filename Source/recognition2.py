# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import imutils

tresh = 100
max_tresh = 255


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
img = cv2.imread(args["image"],0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

ret,thresh1 = cv2.threshold(img,170,255,cv2.THRESH_BINARY);
# Find contours
im2, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#rotrect = cv2.minAreaRect(contours)

# Find the rotated rectangles and ellipses for each contour
minRect = [0]*len(contours)
minEllipse = [0]*len(contours)

for i in range(len(contours)):
    minRect[i] = cv2.minAreaRect( contours[i] )
    if len(contours[i]) > 5:
        minEllipse[i] = cv2.fitEllipse( contours[i] )

# Draw contours + rotated rects + ellipses
# drawing = np.zeros( len(thresh1), cv2.CV_8UC3 )
drawing = img
for i in range(len(contours)):
    color = [255,0,0]
    # contour
    cv2.drawContours( drawing, contours, i, color, 3)
    # ellipse
    # cv2.ellipse( drawing, minEllipse[i], color, 2, 8 )
    # rotated rectangle
    #Point2f rect_points[4]; minRect[i].points( rect_points );
    #for( int j = 0; j < 4; j++ )
    #  line( drawing, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );

img = cv2.imread(args["image"],0)
original_img = cv2.imread(args["image"],0)
img2 = thresh1

skin_row_index = 0
skin_pixel_index = 0
its_skin = False
for row in img:
    skin_pixel_index = 0
    for pixel in row:
        skin_pixel = img2[skin_row_index][skin_pixel_index]
        if skin_pixel != 255:  # WHITE
            img2[skin_row_index][skin_pixel_index] = img[skin_row_index][skin_pixel_index]
        else:
            img2[skin_row_index][skin_pixel_index] = 255
        skin_pixel_index += 1
    skin_row_index += 1

img = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


# noise removal
kernel = np.ones((1,1),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 20)

# sure background area
sure_bg = cv2.dilate(closing,kernel,iterations=20)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,0)

# Threshold
ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,160)


# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

img = img2
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)


circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=30,param2=28,minRadius=10,maxRadius=50)

circles = np.uint16(np.around(circles))
radios = []
resized = imutils.resize(original_img, width=300)
ratio = img.shape[0] / float(resized.shape[0])
for i in circles[0,:]:
    M = cv2.moments(i)
    if M["m00"] == 0:
        M["m00"] = 0.000001
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    # draw the outer circle
    cv2.circle(original_img,(i[0],i[1]),i[2],(0,0,255),2)
    cv2.putText(original_img, 'Circle', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
    radios.append(i[2])
    # draw the center of the circle
    #cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),2)

cv2.imwrite('original_first.png',original_img)

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2]+10,(255,255,255),-2)
    radios.append(i[2])
    # draw the center of the circle
    #cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),2)

height, width, channels = cimg.shape
pix_to_nano_ratio = float(5)/height

print '---------------------HOUGH--TRANSFORM-------------------------'
print 'CIRCULOS INICIALES:',len(circles[0,:])
print 'RADIO PROMEDIO INICIAL:', str(round(sum(radios)/len(radios),2))+' px ('+str(round(float(sum(radios)/len(radios))*pix_to_nano_ratio,2))+' μm)'
cv2.imshow('dets',original_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('first_iteration.png',cimg)
file = open('circulos.txt', 'w')
text = (str(len(circles[0,:])) + '\n' + str(sum(radios)/len(radios)))
file.write(text)
file.close()


