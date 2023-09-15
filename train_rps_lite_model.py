import cv2
import os
import random
import shutil
import tensorflow as tf
import time


IMAGE_SIZE = (96, 96, 3)
LABELS = ["rock", "paper", "scissors"]
TRAINING_DATA_DIR = "data/training"
VALIDATION_DATA_DIR = "data/validation"


random.seed(time.time())


def resize_image(in_image_path, out_image_path):
    in_image = cv2.imread(in_image_path)
    out_image = cv2.resize(in_image, IMAGE_SIZE[0:2])
    cv2.imwrite(out_image_path, out_image)


def preprocess_images(images, src_dir, dst_dir):
    os.makedirs(dst_dir)
    for image in images:
        resize_image(os.path.join(src_dir, image), os.path.join(dst_dir, image))


def preprocess_data():
    # images must be grouped by their label into subdirectories under "data/raw/"
    for label in LABELS:
        raw_dir = os.path.join("data/raw", label)
        images = os.listdir(raw_dir)
        random.shuffle(images)

        # use 80% for training and 20% for validation
        num_training_images = int(0.8 * len(images))
        training_images = images[:num_training_images]
        validation_images = images[num_training_images:]

        preprocess_images(training_images, raw_dir, os.path.join(TRAINING_DATA_DIR, label))
        preprocess_images(validation_images, raw_dir, os.path.join(VALIDATION_DATA_DIR, label))


def build_resnet():
    base_model = tf.keras.applications.ResNet50(include_top = False, input_shape = IMAGE_SIZE, pooling = "avg")
    base_model.trainable = False

    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.Dropout(rate = 0.5))
    model.add(tf.keras.layers.Dense(units = 3, activation = tf.keras.activations.softmax))
    return model


def train_model():
    training_data = tf.keras.utils.image_dataset_from_directory(TRAINING_DATA_DIR,
        image_size = IMAGE_SIZE[0:2],
        label_mode = "categorical",
        class_names = LABELS)
    validation_data = tf.keras.utils.image_dataset_from_directory(VALIDATION_DATA_DIR,
        image_size = IMAGE_SIZE[0:2],
        label_mode = "categorical",
        class_names = LABELS)

    model = build_resnet()
    model.compile(optimizer = tf.keras.optimizers.SGD(), loss = "categorical_crossentropy", metrics = ["accuracy"])
    model.fit(training_data, epochs = 10, validation_data = validation_data)
    return model


def convert_to_lite_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


if __name__ == "__main__":
    preprocess_data()
    model = train_model()
    lite_model = convert_to_lite_model(model)

    with open("rps_model.tflite", "wb") as out:
        out.write(lite_model)
    with open("rps_labels.txt", "wt") as out:
        for label in LABELS:
            out.write(f"{label}\n")

    shutil.rmtree(TRAINING_DATA_DIR)
    shutil.rmtree(VALIDATION_DATA_DIR)

