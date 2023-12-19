import h5py
import tensorflow as tf
from tensorflow.keras.utils import plot_model

model = tf.keras.models.load_model("mushroom_image_classifier_model.h5")
def display_model(model):
    model.summary()
    plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
display_model(model)
