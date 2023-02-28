import cv2
import mediapipe as mp


class FaceDetection:
    def __init__(self, colorize=False, draw_lines=False):
        self.colorize = colorize
        self.draw_lines = draw_lines

        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection()
        self.mpDraw = mp.solutions.drawing_utils

    def draw(self, _img):
        img_rgb = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        results = self.face.process(img_rgb)

        if results.detections:
            for id, detection in enumerate(results.detections):
                if self.draw_lines:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, ic = _img.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(_img, bbox, (0, 106, 255), 2)

    def get_landmark_position(self, hand=0, img=None, draw=False):
        landmarks = []
        results = self.face.process(img)
        if results.multi_hand_landmarks:
            my_hand = results.multi_hand_landmarks[hand]

            for _id, landmark in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmarks.append([_id, cx, cy])

                if draw:
                    self.mpDraw.draw_landmarks(img, my_hand, self.mpFace.HAND_CONNECTIONS)

        return landmarks
