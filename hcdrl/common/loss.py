
import tensorflow as tf


def huber_loss(y_true, y_pred, max_grad=1.0):
    abs_diff = tf.to_float(tf.abs(y_true - y_pred))
    mask = tf.to_float(abs_diff < max_grad / 2)
    return (0.5 * abs_diff**2) * mask + abs_diff * (1 - mask)

def mean_huber_loss(y_true, y_pred, max_grad=1.0):
    return tf.reduce_mean(huber_loss(y_true, y_pred, max_grad))
