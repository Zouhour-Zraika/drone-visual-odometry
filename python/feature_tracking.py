
import cv2


class FeatureTracker:
    def __init__(self, nfeatures=1500):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)

        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=True
        )

    def detect_features(self, image):
        keypoints, descriptors = self.orb.detectAndCompute(
            image, None
        )
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []

        matches = self.matcher.match(desc1, desc2)

        return sorted(
            matches,
            key=lambda m: m.distance
        )
