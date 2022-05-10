## @package main
# Module responsible for handling main site of program
import tensorflow as tf
from yolo_package.utils import DataGenerator, read_annotation_lines
from yolo_package.models import Yolov4
from utils.helpers import ImageUtils
from pathlib import Path
import argparse
import cv2

WEIGHTS_PATH = Path("NN/yolov4.weights")
FOLDER_PATH = Path("ready2learn/images")
CLASS_NAME_PATH = Path("data/classes.txt")
TEST_IMG_PATH = Path("ready2learn/images/DSC06872.JPG")
LABELS_PATH = Path("ready2learn/labels.txt")

gt_folder = Path('fruits_dataset/gt')
pred_folder = Path('fruits_dataset/pred')
temp_folder = Path('fruits_dataset/temp')
output_folder = Path('fruits_dataset/output')


## Function responsible for model train
# @param epochs value of epochs
# @param device type of device
# @param aug if train with augmentation
# @param test_size test size in float number
# @param save_path path to saved model
def train_model(epochs: int, device: str, aug: str, test_size: float, save_path: str):
    IU = ImageUtils()
    IU.resize_db(416, 416)
    if aug == 'YES':
        IU.augment_data()
    print("Prepare process of model training...")
    train_lines, val_lines = read_annotation_lines(
        LABELS_PATH,
        test_size=test_size)

    data_gen_train = DataGenerator(train_lines,
                                   CLASS_NAME_PATH,
                                   FOLDER_PATH)

    data_gen_val = DataGenerator(val_lines,
                                 CLASS_NAME_PATH,
                                 FOLDER_PATH)

    model = Yolov4(weight_path=str(WEIGHTS_PATH),
                   class_name_path=CLASS_NAME_PATH)
    if device == "CPU":
        with tf.device('/CPU:0'):
            model.fit(data_gen_train,
                      initial_epoch=0,
                      epochs=epochs,
                      val_data_gen=data_gen_val,
                      callbacks=[])
    else:
        with tf.device('/device:GPU:0'):
            model.fit(data_gen_train,
                      initial_epoch=0,
                      epochs=epochs,
                      val_data_gen=data_gen_val,
                      callbacks=[])

    print("Saving model...")
    model.save_model(save_path)


## Function responsible for model evaluate
# @param model_path path to model
# @param metrics yes or no for metrics visualization
# @param img_path path to example photo
def evaluate_model(model_path: str, metrics: str, img_path: str):
    print("Loading model...")
    model = Yolov4(weight_path="NN/yolov4.weights",
                   class_name_path=CLASS_NAME_PATH)

    model.load_model(model_path)

    if metrics == 'YES':
        model.export_gt(annotation_path=LABELS_PATH, gt_folder_path=gt_folder)

        model.export_prediction(annotation_path=str(LABELS_PATH), pred_folder_path=pred_folder,
                                img_folder_path=FOLDER_PATH, bs=5)

        model.eval_map(gt_folder_path="fruits_dataset/gt", pred_folder_path='fruits_dataset/pred',
                       temp_json_folder_path='fruits_dataset/temp',
                       output_files_path='fruits_dataset/output')
    img = cv2.imread(filename=str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model.predict_img(img)


parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='command')
train = subparser.add_parser('train')
evaluate = subparser.add_parser('evaluate')

train.add_argument('--epochs', help='number of epochs', type=int, default=10)
train.add_argument('--device', help='device (CPU or CUDA)', type=str, default='cuda:0')
train.add_argument('--save', help='path to save model', type=str, default='model_trained')
train.add_argument('--augmentation', help='add augmentation (YES or NO)', type=str, default='YES')
train.add_argument('--test_size', help='ratio of photos in test set', type=float, default=0.1)

evaluate.add_argument('--model_path', help='path to pretrained model', type=str, required=False,
                      default='trained_500_epochs')
evaluate.add_argument('--metrics', help='show calculated metrics (YES or NO)', type=str, required=False, default='YES')
evaluate.add_argument('--img_path', help='path to test image', type=str, required=False, default=TEST_IMG_PATH)
args = parser.parse_args()

if args.command == 'train':
    print('Run fruit detector')
    train_model(args.epochs, args.device, args.augmentation, args.test_size, args.save)
elif args.command == 'evaluate':
    print('Evaluate model')
    evaluate_model(args.model_path, args.metrics, args.img_path)
    pass
else:
    print('No argument passed')
