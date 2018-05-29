import argparse
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from random import randint
import datetime
import matplotlib
import numpy as np
import math
from time import time
from glob import glob
now = datetime.datetime.now()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', default='../data/kiln/Bangladesh_tiles')
parser.add_argument('--csv_path', default='../data/kiln/bangladesh_results/kiln_idnumbers')
parser.add_argument('--logdir', default='tf_logs')
parser.add_argument('--model_path', default='tmp/new_model.ckpt', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--weight_decay', default=5e-4, type=float)

VGG_MEAN = [123.68, 116.78, 103.94]



def main(args):

    val_filenames = list(glob(args.img_dir + '/*.jpeg'))
    num_classes = 2

    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()

    with graph.as_default():

        def _parse_function(filename, filename2):
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

            return resized_image, filename2

        
        def preprocess(image, latlong):
            crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
            means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            centered_image = crop_image - means                              
            return centered_image, latlong

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
        val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_filenames))
        #val_dataset = val_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
        val_dataset = val_dataset.map(_parse_function,
            num_threads=args.num_workers, output_buffer_size=args.batch_size)
        val_dataset = val_dataset.map(preprocess,
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

        images, idnumbers = iterator.get_next()

        is_training = tf.placeholder(tf.bool)
        
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=False)

        model_path = args.model_path
        assert(os.path.isfile(model_path))

        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)


        saver = tf.train.Saver()
        tf.get_default_graph().finalize()

    with tf.Session(graph=graph) as sess:
        init_fn(sess)
        saver.restore(sess, args.model_path)
        csvfile_path = args.csv_path + now.isoformat() + '.csv'
        csvfile = open(csvfile_path, 'w')
        csvfile.write('latlong,probability of kilnity\n')

        start = time()
        n_images = 0
        n_kilns = 0
        n_nonkilns = 0

        sess.run(val_init_op)
        while True:
            try:


                logits_, idnumbers_ = sess.run([logits, idnumbers], {is_training: False})

                logits_ = np.array(logits_)
                probs = [np.exp(logit) for logit in logits_]
                probs = [y / np.sum(y) for y in probs]
                preds = [np.argmax(y) for y in probs]

                kiln_idx = [i for i, pred in enumerate(preds) if pred == 1]
                if kiln_idx is not None:

                    kiln_probs = [probs[i] for i in kiln_idx]
                    kiln_idnumbers = [idnumbers_[i] for i in kiln_idx]
                    
                    rows = [str(a) + ',' + str(b) + '\n' for a,b in zip(kiln_idnumbers, kiln_probs)]
                    [csvfile.write(row) for row in rows]

                n_images = n_images + len(preds)
                n_kilns = n_kilns + len(kiln_idx)
                n_nonkilns = n_nonkilns + len(preds) - len(kiln_idx)
                print kiln_idnumbers
                print 'number of images seen:     ' + str(n_images)
                print 'number of kilns predicted: ' + str(n_kilns)
                print '-'*50
                print ' '


            except tf.errors.OutOfRangeError:
                break

        
        csvfile.close()

        
        print 'Found ' + str(n_kilns) + ' new kilns.'

        print 'time elapsed: ', time() - start
        


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
