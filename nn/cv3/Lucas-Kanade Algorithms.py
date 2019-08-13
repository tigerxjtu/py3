import cv2
import numpy as np
import matplotlib.pyplot as plt

# some constants and default parameters
lk_params = dict(winSize=(15,15),maxLevel=2,
criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
subpix_params = dict(zeroZone=(-1,-1),winSize=(10,10),
criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,20,0.03))
feature_params = dict(maxCorners=500,qualityLevel=0.01,minDistance=10)

class LKTracker(object):
    """ Class for Lucas-Kanade tracking with
    pyramidal optical flow."""
    def __init__(self,imnames):
        self.imnames = imnames
        self.features = []
        self.tracks = []
        self.current_frame = 0

    def detect_points(self):
        """ Detect ’good features to track’ (corners) in the current frame
        using sub-pixel accuracy. """

        # load the image and create grayscale
        self.image = cv2.imread(self.imnames[self.current_frame])
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # search for good points
        features = cv2.goodFeaturesToTrack(self.gray, **feature_params)
        # refine the corner locations
        cv2.cornerSubPix(self.gray, features, **subpix_params)
        self.features = features
        self.tracks = [[p] for p in features.reshape((-1, 2))]
        self.prev_gray = self.gray

    def track_points(self):
        """ Track the detected features. """

        if self.features != []:
            self.step()  # move to the next frame
            # load the image and create grayscale
            self.image = cv2.imread(self.imnames[self.current_frame])
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # reshape to fit input format
            tmp = np.float32(self.features).reshape(-1, 1, 2)
            # calculate optical flow
            features, status, track_error = cv2.calcOpticalFlowPyrLK(self.prev_gray,self.gray,
                                                                     tmp, None, **lk_params)
            # remove points lost
            self.features = [p for (st, p) in zip(status, features) if st]
            # clean tracks from lost points
            features = np.array(features).reshape((-1, 2))
            for i, f in enumerate(features):
                self.tracks[i].append(f)
            ndx = [i for (i, st) in enumerate(status) if not st]
            ndx.reverse()  # remove from back
            for i in ndx:
                self.tracks.pop(i)
            self.prev_gray = self.gray

    def step(self, framenbr=None):
        """ Step to another frame. If no argument is
        given, step to the next frame. """

        if framenbr is None:
            self.current_frame = (self.current_frame + 1) % len(self.imnames)
        else:
            self.current_frame = framenbr % len(self.imnames)

    def draw(self):
        """ Draw the current image with points using
        OpenCV’s own drawing functions.
        Press ant key to close window."""

        # draw points as green circles
        for point in self.features:
            cv2.circle(self.image, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), -1)
        cv2.imshow('LKtrack', self.image)
        cv2.waitKey()

    def track(self):
        """ Generator for stepping through a sequence."""

        for i in range(len(self.imnames)):
            if self.features == []:
                self.detect_points()
            else:
                self.track_points()
                # create a copy in RGB
            f = np.array(self.features).reshape(-1, 2)
            im = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            yield im, f

imnames = ['Images/Super1.png', 'Images/Super2.png', 'Images/Super3.png']
# create tracker object
# lkt = LKTracker(imnames)
# # detect in first frame, track in the remaining
# lkt.detect_points()
# lkt.draw()
# for i in range(len(imnames)-1):
#     lkt.track_points()
#     lkt.draw()

# track using the LKTracker generator
lkt = LKTracker(imnames)
for im,ft in lkt.track():
    print('tracking %d features' % len(ft))
# plot the tracks
    plt.figure()
    plt.imshow(im)
    for p in ft:
        plt.plot(p[0],p[1],'bo')
    for t in lkt.tracks:
        plt.plot([p[0] for p in t],[p[1] for p in t])
    plt.axis('off')
    plt.show()
