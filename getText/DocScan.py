import cv2
import argparse
import transform
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_local

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="Path to the image")
#ap.add_argument("-c", "--coords", help="Comma separated list of source points")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])  
#image = cv2.imread("scorpionGod.jpeg")
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#pts = np.array(eval(args["coords"]), dtype = "float32")

#pts = np.array([[1227.37, 764.183],[2726.14, 1211.95],
#    [2377.88, 3488.9],[151.491, 2729.37]])

#warp = transform.four_point_transform(image, pts)
#plt.imshow(warp)
#plt.axis("off")
#plt.savefig('warp_'+args["image"])
#plt.show()

(h, w) = image.shape[:2]
ratio = image.shape[0]/500.
orig = image.copy()

height = 500
dim = (int(image.shape[1]/ratio), height)
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

#cv2.imshow("Image", image)
#cv2.imshow("Edged", edged)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
if len(cnts) == 2:
    cnts = cnts[0]
elif len(cnts) == 3:
    cnts = cnts[1]

cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]


# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break

# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
#cv2.imshow("Outline", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

warped = transform.four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
 
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
 
# show the original and scanned images
print("STEP 3: Apply perspective transform")

#cv2.imshow("Original", transform.resize(orig, height=600))
cv2.imshow("Scanned", transform.resize(warped, height=600))
cv2.imwrite( 'scanned.png', transform.resize(warped, height=600))
#cv2.waitKey(0)



