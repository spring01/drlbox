
import tensorflow as tf


class StepCounter:

    def __init__(self):
        self.tf_step_count = tf.Variable(0)

    def set_increment(self):
        ph_inc = tf.placeholder(tf.int32, ())
        self.ph_inc = ph_inc
        self.op_inc = self.tf_step_count.assign_add(ph_inc)

    def set_session(self, sess):
        self.sess = sess

    def increment(self, inc):
        self.sess.run(self.op_inc, feed_dict={self.ph_inc: inc})

    def step_count(self):
        return self.sess.run(self.tf_step_count)
