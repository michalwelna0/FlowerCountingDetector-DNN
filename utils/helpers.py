import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et
from scipy import ndimage
import random
from typing import List, Dict, Tuple

LABEL_PATH = Path("labels")
IMAGE_PATH = Path("photos")
LABEL_PATH_TXT = Path("data/labels.txt")
LABELS_AUGMENTED_PATH = Path("data/labels_aug.txt")
CLASSES_PATH = Path("data/classes.txt")
ALL_LABELS_PATH = Path("ready2learn/labels.txt")
ALL_IMAGES_PATH = Path("ready2learn/images")


class XMLParser(object):
    def __init__(self):
        self.label_path = LABEL_PATH
        self.label_path_txt = LABEL_PATH_TXT
        self.classes_path = CLASSES_PATH
        self.coords: Dict[str, List[Tuple[List[int], str]]] = self._create_annotations()

    @staticmethod
    def _get_coord_from_xml(path: Path) -> List[Tuple[List[int], str]]:
        if path.suffix == '.xml':
            xml_tree = et.parse(path)
            objects = xml_tree.findall("object")
            coords = []
            labels = []
            for obj in objects:
                bbox = obj.find("bndbox")
                name = obj.find("name")
                labels.append(name.text)
                coords.append([int(it.text) for it in bbox])
            return [(coord, label) for coord, label in zip(coords, labels)]

    def _create_annotations(self) -> Dict[str, List[Tuple[List[int], str]]]:
        files = os.listdir(self.label_path)
        print("Creating annotation database...")
        dct = {}
        files = [file for file in files if file.endswith(".xml")]
        for file in files:
            filepath = self.label_path / file
            dct[filepath.stem] = self._get_coord_from_xml(filepath)
        return dct

    def get_classes(self) -> List[str]:
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def create_txt_annotation_file(self) -> None:
        with open(self.label_path_txt, 'w') as f:
            for key, coord_lists in self.coords.items():
                to_write = f'{key}.JPG '
                for lst, classname in coord_lists:
                    converted_list = [str(float(element)) for element in lst]
                    to_write += ','.join(converted_list)
                    to_write += f',{self.get_classes().index(classname)} '
                f.write(to_write[:-1])
                f.write('\n')
        f.close()

    def get_annotation(self, img: str) -> List[List[int]]:
        return [coord for coord, _ in self.coords[img]]


class ImageUtils(object):
    def __init__(self):
        self.image_path = IMAGE_PATH
        self.all_images = ALL_IMAGES_PATH
        self.label_augment = LABELS_AUGMENTED_PATH
        self.all_labels = ALL_LABELS_PATH
        self.last_image_nr = 7070
        self.img_database: Dict[str, np.ndarray] = self._create_db()
        self.parser = XMLParser()
        if not os.path.exists(Path("../ready2learn")):
            os.makedirs(Path("../ready2learn"))
        if not os.path.exists(self.all_images):
            os.makedirs(self.all_images)

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        if path.suffix == '.JPG':
            img = cv2.imread(str(path))
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20, 15))
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    def draw_bboxes(self, img: np.ndarray, file: str) -> None:
        if img.shape[0] > 999:
            thickness = 3
        else:
            thickness = 1
        for xmin, ymin, xmax, ymax in self.parser.get_annotation(file):
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness)

    @staticmethod
    def _rotate_point(points, angle):
        rot_points = []
        ox = 208
        oy = 208
        for i in range(len(points)):
            px = points[i][0]
            py = points[i][1]
            px = np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy) + ox
            py = np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy) + oy
            rot_points.append([int(px), int(py)])
        return rot_points

    def _txt_rotation(self, boxes, angle):
        for i in range(len(boxes)):
            p1 = [boxes[i][0], boxes[i][1]]
            p2 = [boxes[i][2], boxes[i][3]]
            tab_pt = self._rotate_point([p1, p2], angle)
            boxes[i][0:2] = tab_pt[0]
            boxes[i][2:4] = tab_pt[1]
        return boxes

    @staticmethod
    def _txt_flip(boxes, type_of_flip):
        ox = 208
        oy = 208
        for i in range(len(boxes)):
            if type_of_flip == -1:
                boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3] = \
                    2*(ox-boxes[i][2]) + boxes[i][2], 2*(ox-boxes[i][3]) + boxes[i][3], \
                    2*(ox-boxes[i][0]) + boxes[i][0], 2*(ox-boxes[i][1]) + boxes[i][1]
            if type_of_flip == 0:
                boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3] = \
                    boxes[i][0], 2*(ox-boxes[i][3]) + boxes[i][3], boxes[i][2], 2*(ox-boxes[i][1]) + boxes[i][1]
            if type_of_flip == 1:
                boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3] = \
                    2*(ox-boxes[i][2]) + boxes[i][2], boxes[i][1], 2*(ox-boxes[i][0]) + boxes[i][0], boxes[i][3]

        return boxes

    def _read_txt_to_list(self):
        with open(self.parser.label_path_txt) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        return content

    def augment_data(self):
        line = self._read_txt_to_list()
        nr_iter = 0
        list_file = open(self.label_augment, 'w')

        for item in line:
            base_img = str(item.split(".", 1)[0])
            key = "DSC0" + str(self.last_image_nr + nr_iter)
            new_img_name = key + ".JPG"
            list_file.write(new_img_name)
            nr_iter += 1
            item = item.split()
            img_path = item[0]
            boxes = np.array([np.array(list(map(float, box.split(',')))) for box in item[1:]], dtype=np.float32)

            img = self._read_image(self.all_images / img_path)
            type_of_augmentation = random.randint(0, 2)

            # contrast and brightness
            if type_of_augmentation == 0:
                alpha = random.uniform(0.8, 1.2)
                beta = random.uniform(-30, 30)
                img_new = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

            # mirror reflection
            elif type_of_augmentation == 1:
                type_of_flip = random.randint(0, 2) - 1
                img_new = cv2.flip(img, type_of_flip)
                boxes = self._txt_flip(boxes, type_of_flip)

            # rotation
            elif type_of_augmentation == 2:
                angle = random.randint(0, 10) - 5
                img_new = ndimage.rotate(img, -angle, reshape=False)
                boxes = self._txt_rotation(boxes, np.deg2rad(angle))
            else:
                raise ValueError

            path_to_save = self.all_images / new_img_name
            cv2.imwrite(str(path_to_save), img_new)
            coords_update = []
            for i in range(len(boxes)):
                list_file.write(" " + ",".join([str(a) for a in boxes[i][0:4]]) +
                                f',{self.parser.get_classes().index(self.parser.coords[base_img][i][1])}')
                coords_update.append(((boxes[i].astype(int)).tolist()[:4], self.parser.coords[base_img][i][1]))

            # Update database of images and labels
            self.parser.coords[key] = coords_update
            self.img_database[key] = img_new
            list_file.write('\n')

        list_file.close()
        self.last_image_nr += nr_iter

        with open(self.label_augment) as f:
            augmented = f.read()

        with open(self.parser.label_path_txt) as f:
            original = f.read()

        data = original + augmented
        with open(self.all_labels, 'w') as f:
            f.write(data)

    def resize_db(self, new_x: int, new_y: int) -> None:
        print("Resizing database...")
        for (key, image), (_, label) in zip(self.img_database.items(), self.parser.coords.items()):
            old_x, old_y, _ = image.shape
            ratio_x = new_x/old_x
            ratio_y = new_y/old_y
            self.img_database[key] = cv2.resize(image, (new_x, new_y))
            path_to_save = self.all_images / (key + '.JPG')
            cv2.imwrite(str(path_to_save), self.img_database[key])
            for lst, _ in label:
                for i in range(len(lst)):
                    if i % 2 != 0:
                        lst[i] = int(lst[i] * ratio_x)
                    else:
                        lst[i] = int(lst[i] * ratio_y)

        self.parser.create_txt_annotation_file()


if __name__ == '__main__':
    # Example of usage
    # filename = 'DSC06475'
    # Create object of ImageUtilsClass
    IU = ImageUtils()
    # Resize original files into size suitable for NN
    IU.resize_db(416, 416)
    # For debugging purposes you can show bounding boxes on resized image
    # IU.draw_bboxes(IU.img_database[filename], filename)
    # IU.plot_image(IU.img_database[filename])
    # Perform augmentation process
    IU.augment_data()
    '''
    Above instructions save images and labels to directory ready2learn.
    You can now use them in learning process.
    '''
