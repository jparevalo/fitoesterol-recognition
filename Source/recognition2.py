import cv2
import numpy as np

tresh = 100
max_tresh = 255


# Detect edges using Threshold
img = cv2.imread('../Pictures/img2.tif',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

ret,thresh1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY);
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
drawing = np.zeros( len(thresh1) )
for i in range(len(contours)):
    color = cv2.scalar( np.random.uniform(0,255), np.random.uniform(0,255), np.random.uniform(0,255) )
    print color
    # contour
    cv2.drawContours( drawing, contours, i, color, 3)
    # ellipse
    cv2.ellipse( drawing, minEllipse[i], color, 2, 8 )
    # rotated rectangle
    #Point2f rect_points[4]; minRect[i].points( rect_points );
    #for( int j = 0; j < 4; j++ )
    #  line( drawing, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );

# Show in a window
cv2.imshow('detected circles', drawing)
cv2.waitKey(0)
cv2.destroyAllWindows()
