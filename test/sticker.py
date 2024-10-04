import cv2
import numpy as np

class Sticker:
    def __init__(self, image, x_pos, y_pos):
        self.image = image
        self.x_pos = x_pos
        self.y_pos = y_pos

        self.contour = Sticker.get_contour(image)
        self.width, self.height = self.__get_shape()

    def move(self, dx, dy):
        self.x_pos += dx
        self.y_pos += dy

    def rotate(self, angle):
        rotate_matrix = cv2.getRotationMatrix2D(tuple(self.x_pos, self.y_pos), angle, 1.0)

        self.contour = cv2.transform(self.contour, rotate_matrix)

    def __get_shape(self):
        x_coords = self.contour[:, 0]
        y_coords = self.contour[:, 1]

        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)

        return tuple(width, height)

    @staticmethod
    def get_contour(image):
        # Check if the image has an alpha channel (transparency)
        if image.shape[2] == 4:
            alpha_channel = image[:, :, 3]
            _, thresh = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Get contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.vstack(contours)[:, 0, :]

        return contours