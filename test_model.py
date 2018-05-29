import argparse
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from random import randint
import datetime
import matplotlib
from pprint import pprint
import numpy as np
import math
from time import time
now = datetime.datetime.now()
from sklearn.metrics import confusion_matrix, classification_report
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


parser = argparse.ArgumentParser()
parser.add_argument('--val_dir', default='../data/kiln/new_kiln_val')
parser.add_argument('--logdir', default='tf_logs')
parser.add_argument('--model_path', default='kiln_model.ckpt', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--weight_decay', default=5e-4, type=float)

VGG_MEAN = [123.68, 116.78, 103.94]


def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    labels = os.listdir(directory)
    # Sort the labels so that training and validation get them in the same order
    labels.sort()

    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))

    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i

    labels = [label_to_int[l] for l in labels]

    return filenames, labels


def main(args):

    val_filenames, val_labels = list_images(args.val_dir)
    num_classes = len(set(val_labels))

    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()

    with graph.as_default():

        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
            image = tf.cast(image_decoded, tf.float32)

            smallest_side = 256.0
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            height = tf.to_float(height)
            width = tf.to_float(width)

            scale = tf.cond(tf.greater(height, width),
                            lambda: smallest_side / width,
                            lambda: smallest_side / height)
            new_height = tf.to_int32(height * scale)
            new_width = tf.to_int32(width * scale)

            resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
            return resized_image, label

        
        def val_preprocess(image, label):
            crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
            means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            centered_image = crop_image - means                              
            return centered_image, label

        # ----------------------------------------------------------------------
        # DATASET CREATION using tf.contrib.data.Dataset
        # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

        # The tf.contrib.data.Dataset framework uses queues in the background to feed in
        # data to the model.
        # We initialize the dataset with a list of filenames and labels, and then apply
        # the preprocessing functions described above.
        # Behind the scenes, queues will load the filenames, preprocess them with multiple
        # threads and apply the preprocessing in parallel, and then batch the data


        # Validation dataset
        val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        val_dataset = val_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
        val_dataset = val_dataset.map(_parse_function,
            num_threads=args.num_workers, output_buffer_size=args.batch_size)
        val_dataset = val_dataset.map(val_preprocess,
            num_threads=args.num_workers, output_buffer_size=args.batch_size)
        batched_val_dataset = val_dataset.batch(args.batch_size)


        # Now we define an iterator that can operator on either dataset.
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(val_init_op)   for 1 epoch on the valiation set
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.

        # A reinitializable iterator is defined by its structure. We could use the
        # `output_types` and `output_shapes` properties of either `train_dataset`
        # or `validation_dataset` here, because they are compatible.
        iterator = tf.contrib.data.Iterator.from_structure(batched_val_dataset.output_types,
                                                           batched_val_dataset.output_shapes)

        val_init_op = iterator.make_initializer(batched_val_dataset)

        images, labels = iterator.get_next()


        # Indicates whether we are in training or in test mode
        is_training = tf.placeholder(tf.bool)

        # ---------------------------------------------------------------------
        # Now that we have set up the data, it's time to set up the model.
        # For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
        # last fully connected layer (fc8) and replace it with our own, with an
        # output size num_classes=8
        # We will first train the last layer for a few epochs.
        # Then we will train the entire model on our dataset for a few epochs.

        # Get the pretrained model, specifying the num_classes argument to create a new
        # fully connected replacing the last one, called "vgg_16/fc8"
        # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
        # Here, logits gives us directly the predicted scores we wanted from the images.
        # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer
        
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=False)

        # Specify where the model checkpoint is (pretrained weights).
        model_path = args.model_path
        print model_path
        assert(os.path.isfile(model_path))

        # Restore only the layers up to fc7 (included)
        # Calling function `init_fn(sess)` will load all the pretrained weights.
        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        tf.summary.histogram("logits", logits)
        tf.summary.histogram("predictions", prediction)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.image('images', images, max_outputs = 100)
        summary_op = tf.summary.merge_all()


        saver = tf.train.Saver()


        tf.get_default_graph().finalize()

    with tf.Session(graph=graph) as sess:
            init_fn(sess)  # load the pretrained weights
#             val_writer = tf.summary.FileWriter('tf_logs/test', graph=tf.get_default_graph())

            saver.restore(sess, args.model_path)

            start = time()
            sess.run(val_init_op)
            
            preds = []
            labelss = []
            while True:
                try:
                    pred, labels_ = sess.run([prediction, labels], {is_training: False})

                    preds.extend(pred)
                    labelss.extend(labels_)

                except tf.errors.OutOfRangeError:
                    break
                    
            print 'time elapsed: ', time() - start
            print(confusion_matrix(labelss, preds))
            print(classification_report(labelss, preds))
            
#             logits = np.array(logits)
#             prob = [np.exp(y) for y in logits]
#             prob = [y / np.sum(y) for y in prob]
#             prob = [list(y) for y in prob]
#             pprint(zip(range(len(prob)),labels, prob))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
