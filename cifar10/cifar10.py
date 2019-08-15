from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import os
import re
import sys
import tarfile

import tensorflow as tf
import mesh_tensorflow as mtf
from six.moves import urllib
import cifar10_input as dataset

# This gets saved in a folder inside current folder
tf.flags.DEFINE_string("model_dir", "mtf_cifar_model", "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 200,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("hidden_size", 512, "Size of each hidden layer.")
tf.flags.DEFINE_integer("train_epochs", 1, "Total number of training epochs.")
tf.flags.DEFINE_integer("epochs_between_evals", 1,
                        "# of epochs between evaluations.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_string("mesh_shape", "b1:1;b2:1", "mesh shape")
tf.flags.DEFINE_string("layout", "row_blocks:b1;col_blocks:b2",
                       "layout rules")
tf.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

FLAGS = tf.flags.FLAGS


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1 # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001       # Initial learning rate.

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# Normalization of inputs by hand
def bnorm(image):
  image = tf.cast(image, tf.float32)
  image = tf.reshape(image, [FLAGS.batch_size, 32, 32, 3])
  image = image / 255.0  
  return image

def cifar_model(features, labels, mesh):
  """The model.

  Args:
    image: tf.Tensor with shape [batch, 32*32]
    labels: a tf.Tensor with shape [batch] and dtype tf.int32
    mesh: a mtf.Mesh

  Returns:
    logits: a mtf.Tensor with shape [batch, 10]
    loss: a mtf.Tensor with shape []
  """
  features = copy.copy(features)
  batch_dim = mtf.Dimension("batch", FLAGS.batch_size)
  row_blocks_dim = mtf.Dimension("row_blocks", 4)
  col_blocks_dim = mtf.Dimension("col_blocks", 4)
  rows_dim = mtf.Dimension("rows_size", 8)
  cols_dim = mtf.Dimension("cols_size", 8)

  classes_dim = mtf.Dimension("classes", 10)
  one_channel_dim = mtf.Dimension("one_channel", 3)


  image = features['image']
  image = bnorm(image)

  x = mtf.import_tf_tensor(
      mesh, tf.reshape(image, [FLAGS.batch_size, 4, 8, 4, 8, 3]),
      mtf.Shape(
          [batch_dim, row_blocks_dim, rows_dim,
           col_blocks_dim, cols_dim, one_channel_dim]))
  x = mtf.transpose(x, [
      batch_dim, row_blocks_dim, col_blocks_dim,
      rows_dim, cols_dim, one_channel_dim])

  # Add some convolutional layers to demonstrate that convolution works.
  fh_dim = mtf.Dimension("fh", 7)
  fw_dim = mtf.Dimension("fw", 7)
  filters1_dim = mtf.Dimension("filters1", 32)
  filters2_dim = mtf.Dimension("filters2", 32)

  kernel1 = mtf.get_variable(
      mesh, "kernel1", [fh_dim, fw_dim, one_channel_dim, filters1_dim])
  kernel2 = mtf.get_variable(
      mesh, "kernel2", [fh_dim, fw_dim, filters1_dim, filters2_dim])


  f1 = mtf.relu(mtf.conv2d_with_blocks(
      x, kernel1, strides=[1, 1, 1, 1], padding="SAME",
      h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim))


  f2 = mtf.relu(mtf.conv2d_with_blocks(
      f1, kernel2, strides=[1, 1, 1, 1], padding="SAME",
      h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim))


  x = mtf.reduce_mean(f2, reduced_dim=filters2_dim)

  # Add some fully-connected dense layers.
  hidden_dim1 = mtf.Dimension("hidden1", FLAGS.hidden_size)
  hidden_dim2 = mtf.Dimension("hidden2", FLAGS.hidden_size)

  h1 = mtf.layers.dense(
      x, hidden_dim1,
      reduced_dims=x.shape.dims[-4:],
      activation=mtf.relu, name="hidden1")
  h2 = mtf.layers.dense(
      h1, hidden_dim2,
      activation=mtf.relu, name="hidden2")
  logits = mtf.layers.dense(h2, classes_dim, name="logits")

  labels = features['label']
  if labels is None:
    loss = None
  else:
    labels = mtf.import_tf_tensor(
        mesh, tf.reshape(labels, [FLAGS.batch_size]), mtf.Shape([batch_dim]))
    loss = mtf.layers.softmax_cross_entropy_with_logits(
        logits, mtf.one_hot(labels, classes_dim), classes_dim)
    loss = mtf.reduce_mean(loss)

  return logits, loss


def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  tf.logging.info("features = %s labels = %s mode = %s params=%s" %
                  (features, labels, mode, params))
  global_step = tf.train.get_global_step()
  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "my_mesh")
  logits, loss = cifar_model(features, labels, mesh)
  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)
  mesh_size = mesh_shape.size
  # To enable manual device placement (e.g.: GPU 1, 2) comment the line below and uncomment the next one
  mesh_devices = [""] * mesh_size
  # mesh_devices = ['GPU:' + str(i) for i in range(mesh_size)]
  mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
      mesh_shape, layout_rules, mesh_devices)  

  labels = features['label']

  if mode == tf.estimator.ModeKeys.TRAIN:
    var_grads = mtf.gradients(
        [loss], [v.outputs[0] for v in graph.trainable_variables])    
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    mtf_lr = mtf.import_tf_tensor(
        mesh, tf.convert_to_tensor(lr, dtype=tf.float32), mtf.Shape([]))
    optimizer = mtf.optimize.AdafactorOptimizer(learning_rate=mtf_lr)
    update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

  lowering = mtf.Lowering(graph, {mesh: mesh_impl})
  restore_hook = mtf.MtfRestoreHook(lowering)

  tf_logits = lowering.export_to_tf_tensor(logits)
  if mode != tf.estimator.ModeKeys.PREDICT:
    tf_loss = lowering.export_to_tf_tensor(loss)
    tf_loss = tf.to_float(tf_loss)
    tf.summary.scalar("loss", tf_loss)

  if mode == tf.estimator.ModeKeys.TRAIN:
    tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
    tf_update_ops.append(tf.assign_add(global_step, 1))
    train_op = tf.group(tf_update_ops)
    saver = tf.train.Saver(
        tf.global_variables(),
        sharded=True,
        max_to_keep=10,
        keep_checkpoint_every_n_hours=2,
        defer_build=False, save_relative_paths=True)
    tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
    saver_listener = mtf.MtfCheckpointSaverListener(lowering)
    saver_hook = tf.train.CheckpointSaverHook(
        FLAGS.model_dir,
        save_steps=1000,
        saver=saver,
        listeners=[saver_listener])

    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(tf_logits, axis=1))

    # Name tensors to be logged with LoggingTensorHook.
    tf.identity(tf_loss, "cross_entropy")
    tf.identity(accuracy[1], name="train_accuracy")

    # Save accuracy scalar to Tensorboard output.
    tf.summary.scalar("train_accuracy", accuracy[1])

    # restore_hook must come before saver_hook
    return tf.estimator.EstimatorSpec(
        tf.estimator.ModeKeys.TRAIN, loss=tf_loss, train_op=train_op,
        training_chief_hooks=[restore_hook, saver_hook])

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        "classes": tf.argmax(tf_logits, axis=1),
        "probabilities": tf.nn.softmax(tf_logits),
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        prediction_hooks=[restore_hook],
        export_outputs={
            "classify": tf.estimator.export.PredictOutput(predictions)
        })

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=tf_loss,
        evaluation_hooks=[restore_hook],
        eval_metric_ops={
            "accuracy":
            tf.metrics.accuracy(
                labels=labels, predictions=tf.argmax(tf_logits, axis=1)),
        })


def run_cifar():
  """Run cifar training and eval loop."""
  cifar_classifier = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=FLAGS.model_dir)

  # Set up training and evaluation input functions.
  def train_input_fn():
    """Prepare data for training."""

    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes use less memory. MNIST is a small
    # enough dataset that we can easily shuffle the full epoch.
    ds = dataset.train()
    ds = ds.prefetch(-1)
    ds_batched = ds.cache().shuffle(buffer_size=50000).batch(FLAGS.batch_size)

    # Iterate through the dataset a set number (`epochs_between_evals`) of times
    # during each training session.
    ds = ds_batched.repeat(FLAGS.epochs_between_evals)
    return ds

  def eval_input_fn():
    eval_data = 'test'
    return dataset.test(eval_data).batch(
        FLAGS.batch_size).make_one_shot_iterator().get_next()

  # Train and evaluate model
  # TF Timeline: Saves a JSON file for timeline every 100 steps into the specified folder
  train_hooks = tf.train.ProfilerHook(save_steps=100, output_dir=FLAGS.model_dir)
  for _ in range(FLAGS.train_epochs // FLAGS.epochs_between_evals):
    cifar_classifier.train(input_fn=train_input_fn, hooks=None)
    eval_results = cifar_classifier.evaluate(input_fn=eval_input_fn)
    print("\nEvaluation results:\n\t%s\n" % eval_results)


def main(_):
  run_cifar()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
