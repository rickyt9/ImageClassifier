import tensorflow as tf
import numpy as np

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()

def predict_image(image, model, top_k=1):
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    ps = model.predict(processed_image)
    top_probs, labels = tf.math.top_k(ps, k=top_k, sorted=True)
    return top_probs.numpy()[0], labels.numpy()[0]