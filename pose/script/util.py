import math
import numpy as np
import cv2


eps = 0.01

def smart_width(d):
    if d<5:
        return 1
    elif d<10:
        return 2
    elif d<20:
        return 3
    elif d<40:
        return 4
    elif d<80:
        return 5
    elif d<160:
        return 6  
    elif d<320:
        return 7 
    else:
        return 8



def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

            width = smart_width(length)
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), width), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            radius = 4
            cv2.circle(canvas, (int(x), int(y)), radius, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks):
    import matplotlib
    
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    
    # (person_number*2, 21, 2)
    for i in range(len(all_hand_peaks)):
        peaks = all_hand_peaks[i]
        peaks = np.array(peaks)
        
        for ie, e in enumerate(edges):

            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                width = smart_width(length)
                cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=width)

        for _, keyponit in enumerate(peaks):
            x, y = keyponit

            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                radius = 3
                cv2.circle(canvas, (x, y), radius, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks):
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                radius = 3
                cv2.circle(canvas, (x, y), radius, (255, 255, 255), thickness=-1)
    return canvas




# Calculate the resolution 
def size_calculate(h, w, resolution):
    
    H = float(h)
    W = float(w)

    # resize the short edge to the resolution
    k = float(resolution) / min(H, W) # short edge
    H *= k
    W *= k

    # resize to the nearest integer multiple of 64
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    return H, W



def warpAffine_kps(kps, M):
    a = M[:,:2]
    t = M[:,2]
    kps = np.dot(kps, a.T) + t
    return kps




