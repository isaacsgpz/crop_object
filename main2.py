import cv2
import numpy as np


class ObjectCropper:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.processed_image = None

    def preprocess_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    def threshold_image(self, preprocessed_image):
        _, thresholded = cv2.threshold(
            preprocessed_image, 60, 255, cv2.THRESH_BINARY_INV)
        return thresholded

    def apply_morphological_operations(self, thresholded_image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphed = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
        return morphed

    def find_largest_contour(self, morphed_image):
        contours, _ = cv2.findContours(
            morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour

    def crop_object(self, largest_contour):
        mask = np.zeros_like(self.image)
        cv2.drawContours(mask, [largest_contour], -1,
                         (255, 255, 255), thickness=cv2.FILLED)
        result = cv2.bitwise_and(self.image, mask)
        background = np.zeros_like(self.image, dtype=np.uint8)
        self.processed_image = cv2.add(
            result, background, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

    def save_result(self, output_path):
        if self.processed_image is not None:
            cv2.imwrite(output_path, self.processed_image)

    def process(self, output_path):
        preprocessed_image = self.preprocess_image()
        thresholded_image = self.threshold_image(preprocessed_image)
        morphed_image = self.apply_morphological_operations(thresholded_image)
        largest_contour = self.find_largest_contour(morphed_image)
        self.crop_object(largest_contour)
        self.save_result(output_path)


if __name__ == '__main__':
    import os

    IMAGES_DIR = "./images"
    IMAGES_SOURCE = f"{IMAGES_DIR}/source"
    IMAGES_RESULT = f"{IMAGES_DIR}/result"

    for image in os.listdir(IMAGES_SOURCE):
        cropper = ObjectCropper(f"{IMAGES_SOURCE}/{image}")
        cropper.process(f"{IMAGES_RESULT}/{image}")
        print(f"Processed image: {image}")
