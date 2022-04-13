import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et
from typing import List, Dict
LABEL_PATH = Path("../labels")
IMAGE_PATH = Path("../photos")


class XMLParser(object):
    def __init__(self):
        self.label_path = LABEL_PATH
        self.coords: Dict[str, List[List[int]]] = self._create_annotations()

    @staticmethod
    def _get_coord_from_xml(path: Path) -> List[List[int]]:
        if path.suffix == '.xml':
            xml_tree = et.parse(path)
            objects = xml_tree.findall("object")
            coords = []
            for obj in objects:
                bbox = obj.find("bndbox")
                coords.append([int(it.text) for it in bbox])
            return coords

    def _create_annotations(self) -> Dict[str, List[List[int]]]:
        files = os.listdir(self.label_path)
        print("Creating annotation database...")
        dct = {}
        files = [file for file in files if file.endswith(".xml")]
        for file in files:
            filepath = self.label_path / file
            dct[filepath.stem] = self._get_coord_from_xml(filepath)
        return dct

    def get_annotation(self, img: str) -> List[List[int]]:
        return self.coords[img]


class ImageUtils(object):
    def __init__(self):
        self.image_path = IMAGE_PATH
        self.img_database: Dict[str, np.ndarray] = self._create_db()
        self.parser = XMLParser()

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        if path.suffix == '.JPG':
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

    def _create_db(self) -> Dict[str, np.ndarray]:
        files = os.listdir(self.image_path)
        print("Creating image database...")
        files = [file for file in files if file.endswith(".JPG")]
        dct = {}
        for file in files:
            filepath = self.image_path / file
            dct[filepath.stem] = self._read_image(filepath)
        return dct

    @staticmethod
    def plot_image(img: np.ndarray) -> None:
        plt.figure(figsize=(20, 15))
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    def draw_bboxes(self, img: np.ndarray, file: str) -> None:
        for xmin, ymin, xmax, ymax in self.parser.get_annotation(file):
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)

    def resize_db(self, new_x: int, new_y: int) -> None:
        print("Resizing database...")
        for (key, image), (_, label) in zip(self.img_database.items(), self.parser.coords.items()):
            old_x, old_y, _ = image.shape
            ratio_x = new_x/old_x
            ratio_y = new_y/old_y
            self.img_database[key] = cv2.resize(image, (new_x, new_y))
            for lst in label:
                for i in range(len(lst)):
                    if i % 2 != 0:
                        lst[i] = int(lst[i] * ratio_x)
                    else:
                        lst[i] = int(lst[i] * ratio_y)


if __name__ == '__main__':
    # Example of usage
    filename = 'DSC06475'
    IU = ImageUtils()
    IU.resize_db(1024, 1024)
    IU.draw_bboxes(IU.img_database[filename], filename)
    IU.plot_image(IU.img_database[filename])
