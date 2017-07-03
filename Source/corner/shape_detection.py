# import the necessary packages
from shape_detector import ShapeDetector
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred,150,255,cv2.THRESH_BINARY)[1]
cv2.imshow("Image", thresh)
cv2.waitKey(0)

# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
                        cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if imutils.is_cv2() else cnts[1]
sd = ShapeDetector()
# loop over the contours
ci = 0
b = 0
radios = []
for c in cnts[:len(cnts)-1]:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    if M["m00"] == 0:
        M["m00"] = 0.000001
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape, radio = sd.detect(c)
    if shape != "pentagon":
        if shape == 'Circulo':
            ci += 1
            radios.append(radio)
        elif shape == 'Baston':
            b += 1
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
print '--------------------------------------------------------------'
print 'CIRCULOSSEGUNDA ETAPA: ', ci
print 'RADIO PROMEDIO SEGUNDA ETAPA: ', sum(radios)/len(radios)
print '--------------------------------------------------------------'
print 'BASTONES', b
