import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable

"""
本文件对原 particle_model.py 做了以下关键变更（不修改数据生成脚本 Particle_Tracking_Training_Data.py）：
1) 仍然输出 (128, 128, 2) 的 one-hot 两通道 softmax，保持与现有标签完全对齐。
2) 去掉 300× 类权重与 'accuracy' 指标；改用更适合极度不均衡分割的：
   - 损失：CategoricalCrossentropy (CE) + λ*(1 - Dice_fg)（默认 λ=0.5）
   - 指标：Dice_fg、IoU_fg、PR_AUC_fg（正类前景）
   这样不需要夸张的 class weight 也能稳定地优化前景像素。
3) 显式设置优化器与学习率（Adam(1e-3)），并提供可切换到 focal loss 的接口（可选）。
4) 所有自定义损失/指标均用 @register_keras_serializable 装饰，便于保存/加载。

使用方法：
from particle_model import particle_tracking_model, compile_particle_model
m = particle_tracking_model(input_shape=(256,256,1), num_classes=2)
m = compile_particle_model(m, optimizer=tf.keras.optimizers.Adam(1e-3),
                           loss_name='ce_dice', dice_weight=0.5)
# 之后直接 fit(...)，不需要再传 300× 的 class weight。
"""


# -----------------------------
# Model
# -----------------------------

def particle_tracking_model(input_shape=(256, 256, 1), num_classes=2):
    """构建输出 (128, 128, num_classes) 的分割模型。
    说明：只做一次 2× 下采样以匹配你现有的 128×128 标签。
    如后续要升级为 U-Net，可在不改标签的前提下替换此骨干。
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder（一次下采样，得到 128×128）
    x = layers.Conv2D(32, 3, padding='same', activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 256->128
    x = layers.Dropout(0.25)(x)

    # Feature blocks（保持 128×128）
    x = layers.Conv2D(64, 3, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)

    # 输出层：两通道 softmax，对应 [背景, 前景]
    outputs = layers.Conv2D(num_classes, kernel_size=1, activation='softmax', name='segmentation')(x)
    return models.Model(inputs, outputs, name='particle_tracking_model')


# -----------------------------
# Losses & Metrics（前景 = 通道 index 1）
# -----------------------------

@register_keras_serializable()
def dice_fg(y_true, y_pred, smooth: float = 1e-6):
    """前景通道（index=1）的 Dice 系数（连续概率版）。
    兼容 y_true 为整型的情况，将其投射为与 y_pred 相同的 dtype。"""
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_fg = tf.reshape(y_true[..., 1], [-1])
    y_pred_fg = tf.reshape(y_pred[..., 1], [-1])
    intersection = tf.reduce_sum(y_true_fg * y_pred_fg)
    denom = tf.reduce_sum(y_true_fg) + tf.reduce_sum(y_pred_fg)
    return (2.0 * intersection + smooth) / (denom + smooth)

@register_keras_serializable()
def iou_fg(y_true, y_pred, threshold: float = 0.5, smooth: float = 1e-6):
    """前景通道 IoU（将 y_pred 概率阈值化）。
    兼容 y_true 为整型的情况，将其投射为与 y_pred 相同的 dtype。"""
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_fg = tf.reshape(y_true[..., 1], [-1])
    y_pred_bin = tf.cast(y_pred[..., 1] > threshold, tf.float32)
    y_pred_bin = tf.reshape(y_pred_bin, [-1])
    intersection = tf.reduce_sum(y_true_fg * y_pred_bin)
    union = tf.reduce_sum(y_true_fg) + tf.reduce_sum(y_pred_bin) - intersection
    return (intersection + smooth) / (union + smooth)

@register_keras_serializable()
class PRAUCForeground(tf.keras.metrics.Metric):
    """前景通道的 PR-AUC（适合极不均衡）。"""
    def __init__(self, name='pr_auc_fg', **kwargs):
        super().__init__(name=name, **kwargs)
        self._auc = tf.keras.metrics.AUC(curve='PR', name='auc_pr_internal')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_fg = tf.reshape(y_true[..., 1], [-1])
        y_pred_fg = tf.reshape(y_pred[..., 1], [-1])
        self._auc.update_state(y_true_fg, y_pred_fg, sample_weight)

    def result(self):
        return self._auc.result()

    def reset_states(self):
        self._auc.reset_states()


@register_keras_serializable()
def ce_plus_dice_loss(dice_weight: float = 0.5):
    """CE + λ*(1 - Dice_fg)。dice_weight 可调，0.3~1.0 常见。
    说明：CE 负责概率校准与整体分类，Dice 强化稀疏前景的重合度。
    """
    ce = tf.keras.losses.CategoricalCrossentropy()

    def _loss(y_true, y_pred):
        ce_val = ce(y_true, y_pred)
        dice_val = dice_fg(y_true, y_pred)
        return ce_val + dice_weight * (1.0 - dice_val)

    return _loss


@register_keras_serializable()
def categorical_focal_loss(gamma: float = 2.0, alpha_bg: float = 0.5, alpha_fg: float = 0.5):
    """可选：分类 Focal Loss（多类版，两通道）。默认不再使用极端类权重。
    - gamma 控制易分类样本的衰减；
    - alpha_* 为类别平衡系数，可轻微偏向前景，如 alpha_fg=0.7。
    """
    def _loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)  # per-pixel CE
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)               # 选择正确类的概率
        alpha_t = y_true[..., 0] * alpha_bg + y_true[..., 1] * alpha_fg
        fl = alpha_t * tf.pow(1.0 - p_t, gamma) * ce
        return tf.reduce_mean(fl)

    return _loss


# -----------------------------
# Compile / Load helpers
# -----------------------------

def _resolve_loss(loss_name: str, dice_weight: float):
    loss_name = (loss_name or 'ce_dice').lower()
    if loss_name in ['ce_dice', 'dice', 'ce+dice']:
        return ce_plus_dice_loss(dice_weight=dice_weight)
    if loss_name in ['focal', 'categorical_focal']:
        return categorical_focal_loss(gamma=2.0, alpha_bg=0.5, alpha_fg=0.5)
    if loss_name in ['ce', 'cce', 'categorical_crossentropy']:
        return tf.keras.losses.CategoricalCrossentropy()
    raise ValueError(f"Unknown loss_name: {loss_name}")


def compile_particle_model(model: tf.keras.Model,
                           optimizer=None,
                           loss_name: str = 'ce_dice',
                           dice_weight: float = 0.5,
                           learning_rate: float | None = None,
                           include_accuracy: bool = False) -> tf.keras.Model:
    """为模型进行编译：默认使用 CE+Dice；指标为 Dice/IoU/PR-AUC。
    - include_accuracy=False：不默认暴露 accuracy（在不均衡下可误导）。
    - 如需 focal，可传 loss_name='focal'（不需要极端类权重）。
    """
    if optimizer is None:
        lr = 1e-3 if learning_rate is None else learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    loss_fn = _resolve_loss(loss_name, dice_weight=dice_weight)

    metrics = [dice_fg, iou_fg, PRAUCForeground(name='pr_auc_fg')]
    if include_accuracy:
        metrics.append('accuracy')  # 可选，仅用于调试

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    return model


def load_particle_model(path: str | bytes | tf.io.gfile.GFile):
    """加载使用本模块自定义对象保存的模型。"""
    return tf.keras.models.load_model(
        path,
        custom_objects={
            'dice_fg': dice_fg,
            'iou_fg': iou_fg,
            'PRAUCForeground': PRAUCForeground,
            'ce_plus_dice_loss': ce_plus_dice_loss,
            'categorical_focal_loss': categorical_focal_loss,
        },
        compile=True,
    )
