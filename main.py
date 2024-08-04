import cv2
import numpy as np


class ObjectCropper:
    def __init__(self, source_path, result_path):
        self.source_path = source_path
        self.result_path = result_path

    def _pre_process_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def _threshold_image(self, preprocessed_image):
        _, result = cv2.threshold(
            preprocessed_image, 85, 255, cv2.THRESH_BINARY_INV)
        return result

    def _apply_morphological_operations(self, thresh):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        erode = cv2.dilate(thresh, kernel, iterations=1)
        morphed = cv2.morphologyEx(erode, cv2.MORPH_ERODE, kernel)
        return morphed

    def _find_largest_contour(self, morphed_image):
        contours, _ = cv2.findContours(
            morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea)

    def _crop_object(self, image, largest_contour):
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [largest_contour], -1,
                         (255, 255, 255), thickness=cv2.FILLED)
        result = cv2.bitwise_and(image, mask)
        background = np.zeros_like(image, dtype=np.uint8)
        cv2.add(result, background, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
        return result

    def _save_result(self, image_file, processed_image):
        path = f"{self.result_path}/{image_file}"
        cv2.imwrite(path, processed_image)
        return path

    def _process(self, image_file):
        self.image = cv2.imread(f"{self.source_path}/{image_file}")

        preprocessed_image = self._pre_process_image(self.image)
        thresh_image = self._threshold_image(preprocessed_image)
        morphed_image = self._apply_morphological_operations(thresh_image)
        largest_contour = self._find_largest_contour(morphed_image)

        cropped_object = self._crop_object(self.image, largest_contour)

        return self._save_result(image_file, cropped_object)

    def read_and_process_images(self):
        for image in os.listdir(self.source_path):
            path = self._process(image)
            print(f"Processed image saved at: {path}")


if __name__ == '__main__':
    import os

    IMAGES_DIR = "./images"
    IMAGES_SOURCE = f"{IMAGES_DIR}/source"
    IMAGES_RESULT = f"{IMAGES_DIR}/result"

    cropper = ObjectCropper(source_path=IMAGES_SOURCE,
                            result_path=IMAGES_RESULT)
    cropper.read_and_process_images()
