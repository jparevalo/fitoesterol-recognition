# -*- coding: utf-8 -*-

# import the necessary packages
from shape_detector import ShapeDetector
import argparse
import numpy as np
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-o", "--original", required=True,
                help="path to the original input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
original_image = cv2.imread(args["original"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred,150,255,cv2.THRESH_BINARY)[1]

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
        cv2.drawContours(original_image, [c], -1, (0, 255, 0), 2)
        cv2.putText(original_image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

        # show the output image
        cv2.imshow("Image", original_image)
        cv2.waitKey(0)

height, width, channels = image.shape
pix_to_nano_ratio = float(5)/height

ra = float(sum(radios))/len(radios)
print '--------------------SHAPE--DETECTION--------------------------'
print 'CIRCULOS SEGUNDA ETAPA: ', ci
print 'RADIO PROMEDIO SEGUNDA ETAPA: ', str(round(ra,2))+' px ('+str(round(float(ra)*pix_to_nano_ratio,2))+' μm)'
print 'BASTONES: ', b

file = open('../circulos.txt', 'r')
text = file.read().split()
circulos = int(text[0].strip())
radios = int(text[1].strip())
file.close()

circulos_finales = ci + circulos
radios_finales = ra*(float(ci)/circulos_finales) + radios*(float(circulos)/circulos_finales)


cv2.imshow("Image", original_image)

print '-------------------PRELIMINARY--RESULTS-----------------------'
print 'CIRCULOS TOTALES: ', circulos_finales
print 'RADIO PROMEDIO: ', str(round(radios_finales,2))+' px ('+str(round(float(radios_finales)*pix_to_nano_ratio,2))+' μm)'
print 'BASTONES: ', b
print 'RATIO (#bastones/#circulos)', str(round(100*float(b)/circulos_finales,2))+'%'
print 'DENSIDAD DE CIRCULOS EN LA IMAGEN: ', str(round(100*circulos_finales*(2*np.pi*(radios_finales**2))/(height*width),2)) + '%'


print '-----------------------ADJUSTMENTS----------------------------'
remove_cane = int(raw_input('Ingrese el número de bastones que no se considerarán: '))
add_cane = int(raw_input('Ingrese el número de bastones a agregar: '))
remove_circle = int(raw_input('Ingrese el número de circulos que no se considerarán: '))
add_circle = int(raw_input('Ingrese el número de circulos a agregar: '))

circulos_finales = circulos_finales - remove_circle + add_circle
b = b - remove_cane + add_cane

print '----------------------FINAL_RESULTS---------------------------'
print 'CIRCULOS TOTALES: ', circulos_finales
print 'RADIO PROMEDIO: ', str(round(radios_finales,2))+' px ('+str(round(float(radios_finales)*pix_to_nano_ratio,2))+' μm)'
print 'BASTONES: ', b
print 'RATIO (#bastones/#circulos)', str(round(100*float(b)/circulos_finales,2))+'%'
print 'DENSIDAD DE CIRCULOS EN LA IMAGEN: ', str(round(100*circulos_finales*(2*np.pi*(radios_finales**2))/(height*width),2)) + '%'
print '--------------------------------------------------------------'
raw_input('Presione cualquier tecla para salir...')

cv2.destroyAllWindows()


