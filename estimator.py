import numpy as np
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnGRU, CudnnRNNRelu, CudnnRNNTanh, CudnnLSTM
import ray
from ray import tune
from ray.tune.median_stopping_rule import MedianStoppingRule
import random
def input_fn(batch_size, train=True, put_label=True):
    x = np.float32(np.random.uniform(low=-0.5, high=0.5, size=[32, 100]))

    label = np.int32(np.random.uniform(low=0, high=1.3, size=[32]))

    dataset = tf.data.Dataset.from_tensor_slices((x, label) if put_label else x)

    if train:
        return dataset.shuffle(1000).repeat().batch(batch_size)
    else:
        return dataset.batch(batch_size)


class Trainer(tune.Trainable):
    def model_fn(self, features, labels, mode):
        activation_fn = getattr(tf.nn, self.config['activation'])
        logits = tf.layers.dense(features, self.config["num_units"], activation=activation_fn)
        logits = tf.layers.dense(logits, 2)

        predicted_classes = tf.argmax(logits, 1)
        optimizer = tf.train.AdamOptimizer(self.config["lr"])
        acc = tf.metrics.accuracy(predictions=predicted_classes, labels=labels)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        metrics = {"accuracy": acc}

        if mode == tf.estimator.ModeKeys.PREDICT:
            result = {
                'class_ids': predicted_classes,
                'probabilities': tf.nn.softmax(logits),
                'logits': logits
            }
            return tf.estimator.EstimatorSpec(mode, predictions=result)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        else:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def _setup(self):
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn)
        self.iteration = 0

    def _train(self):
        self.estimator.train(input_fn=lambda: input_fn(15), hooks=[], steps=10)
        eval_result = self.estimator.evaluate(input_fn=lambda: input_fn(15, train=False, put_label=True))
        self.iteration += 1
        return tune.TrainingResult(mean_validation_accuracy=eval_result["accuracy"], timesteps_this_iter=10)

#
# def main():
#     tf.logging.set_verbosity(tf.logging.INFO)
#
#     estimator = tf.estimator.Estimator(model_fn=model_fn)
#     train_result = estimator.train(input_fn=lambda: input_fn(15), hooks=[], steps=100)
#     eval_result = estimator.evaluate(input_fn=lambda: input_fn(15, train=False, put_label=True))
#     estimator.predict(input_fn=lambda: input_fn(15, train=False, put_label=False))
#
#     print(train_result)
#     print(eval_result)