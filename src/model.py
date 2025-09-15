# --------------------------
# Models
# --------------------------
import tensorflow as tf
from tensorflow import keras

# Alias layers without importing tensorflow.keras directly (improves import resolution)
L = keras.layers
def DoubleConv(c):
    return keras.Sequential([
        L.Conv2D(c, 3, padding="same"),
        L.BatchNormalization(),
        L.ReLU(),
        L.Conv2D(c, 3, padding="same"),
        L.BatchNormalization(),
        L.ReLU(),
    ])

def build_unet(input_shape=(256,256,1)):
    x_in = L.Input(shape=input_shape)
    # encoder
    x1 = DoubleConv(32)(x_in); p1 = L.MaxPool2D()(x1)
    x2 = DoubleConv(64)(p1);   p2 = L.MaxPool2D()(x2)
    x3 = DoubleConv(128)(p2);  p3 = L.MaxPool2D()(x3)
    x4 = DoubleConv(256)(p3)
    # decoder
    u3 = L.UpSampling2D()(x4); u3 = L.Concatenate()([u3, x3]); u3 = DoubleConv(128)(u3)
    u2 = L.UpSampling2D()(u3); u2 = L.Concatenate()([u2, x2]); u2 = DoubleConv(64)(u2)
    u1 = L.UpSampling2D()(u2); u1 = L.Concatenate()([u1, x1]); u1 = DoubleConv(32)(u1)
    out = L.Conv2D(1, 1, activation="sigmoid", name="heatmap")(u1)
    model = keras.Model(x_in, out, name="HeatmapUNet")
    return model

def build_z_regressor(patch_size=21):
    x_in = L.Input(shape=(patch_size, patch_size, 1))
    x = L.Conv2D(32, 3, padding="same", activation="relu")(x_in)
    x = L.MaxPool2D()(x)
    x = L.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = L.MaxPool2D()(x)
    x = L.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(128, activation="relu")(x)
    x = L.Dropout(0.2)(x)
    out = L.Dense(1, activation=None, name="z")(x)  # linear, we will train on normalized z
    return keras.Model(x_in, out, name="ZRegressor")

# --------------------------
# Losses & metrics
# --------------------------
def dice_loss(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - dice

class HeatmapLoss(keras.losses.Loss):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, name="HeatmapLoss"):
        super().__init__(name=name)
        self.bce = keras.losses.BinaryCrossentropy()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def call(self, y_true, y_pred):
        return self.bce_weight * self.bce(y_true, y_pred) + \
               self.dice_weight * dice_loss(y_true, y_pred)