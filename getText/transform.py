import cv2
import numpy as np

def resize(image, height):
    (h, w) = image.shape[:2]
    ratio = image.shape[0]/height
    orig = image.copy()

    dim = (int(image.shape[1]/ratio), height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image

def order_points(pts):
    # insert a list od coordinates that will be ordered
    # such that the first of the list is the top-left, 
    # the second is the top-right, the third is the 
    # bottom-right and the fourth is the bottom-left.
    rect = np.zeros((4,2), dtype="float32")
    
    # The top-left corner will have the smallest x+y
    # while the bottom-right the largest.
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # Obtain a consistent order of the four points
    # and unpack them individually.
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Width of new image
    w1 = np.sqrt((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2)
    w2 = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)
    maxW = max(np.int(w1), np.int(w2))

    # Height of the new image
    h1 = np.sqrt((tl[0]-bl[0])**2 + (tl[1]-bl[1])**2)
    h2 = np.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2)
    maxH = max(np.int(h1), np.int(h2))

    # Destination points
    dst = np.array([[0, 0],[maxW-1, 0],
          [maxW-1, maxH-1],[0, maxH-1]],
          dtype = "float32")

    # Transformation matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # Apply M
    warped = cv2.warpPerspective(image, M, (maxW, maxH))

    return warped









