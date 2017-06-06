import cv2
import numpy as np

tresh = 100
max_tresh = 255


# Detect edges using Threshold
img = cv2.imread('../Pictures/img3.tif',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

ret,thresh1 = cv2.threshold(img,170,255,cv2.THRESH_BINARY);
# Find contours
im2, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#rotrect = cv2.minAreaRect(contours)

# Find the rotated rectangles and ellipses for each contour
minRect = [0]*len(contours)
minEllipse = [0]*len(contours)

print minRect, minEllipse

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

img = cv2.imread('../Pictures/img3.tif',0)
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



# Show in a window
cv2.imshow('detected circles', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
