import tensorflow as tf

EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

def load_model(model_path):
    return tf.keras.models.load_model(model_path)
