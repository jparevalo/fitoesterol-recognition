import numpy as np
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

img = cv2.imread(args["image"],0)
filter_size = (5 , 5)
f = np.ones(filter_size)

def media(image, f, idx, idy, out):
        aux = []
        for x in range(len(f)):
                for y in range(len(f[x])):
                       aux.append(image[idx + x][idy + y])
        size = len(aux)
	aux.sort()
        pixel_position = size/2
        out[idx+1][idy+1] = aux[pixel_position]


def fullMedia(image, f):
        
	out = []
	for x in img:
        	out.append(x)
	out = np.array(out)

        x_steps = len(image) - len(f) + 1
        y_steps = len(image[0])  - len(f) + 1

        for x in range(x_steps):
                for y in range(y_steps):
                        media(image,f, x, y, out )
	return out
                        
new_image = fullMedia(img, f)
seccond_image = fullMedia(new_image, f)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.imshow('image',new_image)
cv2.waitKey(0)
cv2.imshow('image',seccond_image)
cv2.imwrite('Seccond_iteration_salt_and_peper.png',seccond_image)
cv2.waitKey(0)


cv2.destroyAllWindows()
