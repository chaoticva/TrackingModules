import cv2
import mediapipe as mp


class PoseDetection:
    def __init__(self, colorize=False, draw_lines=False):
        self.colorize = colorize
        self.draw_lines = draw_lines

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    def draw(self, _img):
        img_rgb = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
                if self.draw_lines:
                    self.mpDraw.draw_landmarks(_img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

                for _id, landmark in enumerate(results.pose_landmarks.landmark):
                    h, w, c = _img.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    if self.colorize:
                        if _id in (4, 8, 12, 16, 20):
                            cv2.circle(_img, (cx, cy), 5, (0, 106, 255), cv2.FILLED)
                        if _id in (3, 7, 11, 15, 19):
                            cv2.circle(_img, (cx, cy), 5, (0, 106, 100), cv2.FILLED)
                        if _id in (2, 6, 10, 14, 18):
                            cv2.circle(_img, (cx, cy), 5, (100, 106, 255), cv2.FILLED)
                        if _id in (1, 5, 9, 13, 17):
                            cv2.circle(_img, (cx, cy), 5, (100, 106, 0), cv2.FILLED)
                        if _id == 0:
                            cv2.circle(_img, (cx, cy), 5, (0, 255, 255), cv2.FILLED)

    def get_landmark_position(self, hand=0, img=None, draw=False):
        landmarks = []
        results = self.pose.process(img)
        if results.multi_hand_landmarks:
            my_hand = results.multi_hand_landmarks[hand]

            for _id, landmark in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmarks.append([_id, cx, cy])

                if draw:
                    self.mpDraw.draw_landmarks(img, my_hand, self.mpPose.HAND_CONNECTIONS)

        return landmarks
