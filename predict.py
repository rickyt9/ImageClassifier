
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image
from model_utils import predict_image

parser = argparse.ArgumentParser(description='Predicts the type of flower of the image provided')
parser.add_argument('image_path', help='Path of the image file')
parser.add_argument('saved_model', help='Path to the saved network model')
parser.add_argument('--top_k' , type=int, help='Returns the top K most likely classes')
parser.add_argument('--category_names', help='Path to a JSON file mapping labels to flower names')

args = parser.parse_args()

model = tf.keras.models.load_model(args.saved_model,
                                   custom_objects={'KerasLayer':hub.KerasLayer},
                                   compile=False)

image = np.asarray(Image.open(args.image_path))
top_k = 1 if args.top_k is None else args.top_k
category_names = args.category_names

probs, classes = predict_image(image, model, top_k)
number_classes = [str(label + 1) for label in classes]
labels = number_classes

if category_names is not None:
    with open(category_names, 'r') as f: class_names = json.load(f)
    named_classes = [class_names[label] for label in number_classes]
    labels = named_classes

    
print('-'*40)
print('{:<4s}{:<30s}{:<10s}'.format('#', 'FLOWER', 'PROBABILITY'))
print('-'*40)
for i in range(top_k):
    print('{:<4d}{:<30s}{:<10f}'.format(i + 1, labels[i], probs[i]))
