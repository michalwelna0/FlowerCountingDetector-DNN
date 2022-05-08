from yolo_package.utils import DataGenerator, read_annotation_lines
from yolo_package.models import Yolov4
from utils.helpers import ImageUtils
from pathlib import Path
import argparse

WEIGHTS_PATH = Path("NN/yolov4.weights")
FOLDER_PATH = Path("ready2learn/images")
CLASS_NAME_PATH = Path("data/classes.txt")
TEST_IMG_PATH = Path("ready2learn/images/DSC06872.JPG")
LABELS_PATH = Path("ready2learn/labels.txt")


def train_model(epochs, aug, test_size, save_path):
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

    model.fit(data_gen_train,
              initial_epoch=0,
              epochs=epochs,
              val_data_gen=data_gen_val,
              callbacks=[])

    print("Saving model...")
    model.save_model(save_path)


def evaluate_model(args):
    print("Loading model...")


parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='command')
train = subparser.add_parser('train')
evaluate = subparser.add_parser('evaluate')

train.add_argument('--epochs', help='number of epochs', type=int, default=10)
train.add_argument('--device', help='device (CPU or CUDA)', type=str, default='cuda:0')
train.add_argument('--save', help='path to save model', type=str, default='model_trained')
train.add_argument('--augmentation', help='add augmentation (YES or NO)', type=str, default='YES')
train.add_argument('--test_size', help='ratio of photos in test set', type=float, default=0.1)

evaluate.add_argument('--model_path', help='path to pretrained model', type=str, required=True)
evaluate.add_argument('--metrics', help='show calculated metrics (YES or NO)', type=str, default='YES')
evaluate.add_argument('--img_path', help='path to test image', type=str, default=TEST_IMG_PATH)
args = parser.parse_args()

if args.command == 'train':
    print('Run fruit detector')
    train_model(args.epochs, args.augmentation, args.test_size, args.save)
elif args.command == 'evaluate':
    print('Evaluate model')
    # evaluate_model(args)
    pass
else:
    print('No argument passed')

# from yolo_package.utils import DataGenerator, read_annotation_lines
# from yolo_package.models import Yolov4
# from pathlib import Path
#
# train_lines, val_lines = read_annotation_lines(
#     Path("ready2learn/labels.txt"),
#     test_size=0.1)
#
# FOLDER_PATH = Path("ready2learn/images")
# class_name_path = Path("data/classes.txt")
#
# data_gen_train = DataGenerator(train_lines,
#                                class_name_path,
#                                FOLDER_PATH)
# data_gen_val = DataGenerator(val_lines,
#                              class_name_path,
#                              FOLDER_PATH)
#
# model = Yolov4(weight_path="NN/yolov4.weights",
#                class_name_path=class_name_path)
#
# model.fit(data_gen_train,
#           initial_epoch=0,
#           epochs=2,
#           val_data_gen=data_gen_val,
#           callbacks=[])
#
# model.save_model("trained_2_epochs")