import mediapipe as mp
import cv2


class FaceMesh:
    mp_draw = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=[0, 106, 255])

    def __init__(self, max_faces, flag: int):
        self.flag = flag
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=max_faces)

    def draw(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(img_rgb)
        if result.multi_face_landmarks:
            for landmark in result.multi_face_landmarks:
                if self.flag == 0:
                    self.mp_draw.draw_landmarks(img, landmark, self.mp_face_mesh.FACEMESH_CONTOURS, self.draw_spec)
                if self.flag == 1:
                    self.mp_draw.draw_landmarks(img, landmark, self.mp_face_mesh.FACEMESH_TESSELATION, self.draw_spec)
