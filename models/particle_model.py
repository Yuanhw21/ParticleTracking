import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable

def particle_tracking_model(input_shape=(256, 256, 1), num_classes=2):
    """Build the segmentation model that outputs (128, 128, 2)."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.UpSampling2D((2, 2)),
        layers.Dropout(0.5),

        layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Conv2D(num_classes, (1, 1), activation='softmax', padding='same'),
    ])
    return model

@register_keras_serializable()
def weighted_binary_crossentropy(y_true, y_pred):
    """Weighted binary crossentropy on 2-class softmax output."""
    weight_for_0 = 1.0
    weight_for_1 = 300.0
    # Per-pixel weights: use class-1 weight where ground truth class-1 is 1
    weights = tf.where(tf.equal(y_true[..., 1], 1), weight_for_1, weight_for_0)
    # Binary crossentropy across the last axis
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weighted_bce = weights * bce
    return tf.reduce_mean(weighted_bce)

def compile_particle_model(model, optimizer='adam'):
    """Compile model with the custom loss and accuracy metric."""
    model.compile(optimizer=optimizer,
                  loss=weighted_binary_crossentropy,
                  metrics=['accuracy'])
    return model

def load_particle_model(path):
    """Load a model saved with this module's custom loss."""
    return tf.keras.models.load_model(
        path,
        custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy}
    )