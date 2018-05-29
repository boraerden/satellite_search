"""
Download the weights trained on ImageNet for VGG:
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
rm vgg_16_2016_08_28.tar.gz
"""

import argparse
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from random import randint
import datetime
from tensorflow.core.protobuf import saver_pb2
import numpy as np
from pprint import pprint
from pdb import set_trace as t
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 



now = datetime.datetime.now()


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='../data/kiln/new_kiln')
parser.add_argument('--val_dir', default='../data/kiln/new_kiln_val')
parser.add_argument('--logdir', default='tf_logs')
parser.add_argument('--model_path', default='vgg_16.ckpt', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=10, type=int)
parser.add_argument('--num_epochs2', default=10, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=5e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.4, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--save_if_train_acc_above', default=0.97, type=float)
parser.add_argument('--save_if_valid_acc_above', default=0.95, type=float)

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
    # Get the list of filenames and corresponding list of labels for training et validation
    train_filenames, train_labels = list_images(args.train_dir)
    val_filenames, val_labels = list_images(args.val_dir)
    assert set(train_labels) == set(val_labels),\
           "Train and val labels don't correspond:\n{}\n{}".format(set(train_labels),
                                                                   set(val_labels))

    num_classes = len(set(train_labels))

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

        def training_preprocess(image, label):
            image = tf.random_crop(image, [224, 224, 3])                       
            image = tf.image.random_flip_left_right(image)                
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
            image = tf.image.random_hue(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
            means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            image = image - means
            return image, label

        
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

        # Training dataset
        train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
        train_dataset = train_dataset.map(_parse_function,
            num_threads=args.num_workers, output_buffer_size=args.batch_size)
        train_dataset = train_dataset.map(training_preprocess,
            num_threads=args.num_workers, output_buffer_size=args.batch_size)
        batched_train_dataset = train_dataset.batch(args.batch_size)

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
        iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                           batched_train_dataset.output_shapes)
        images, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)

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
            logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=is_training,
                                   dropout_keep_prob=args.dropout_keep_prob)

        # Specify where the model checkpoint is (pretrained weights).
        model_path = args.model_path
        assert(os.path.isfile(model_path))

        # Restore only the layers up to fc7 (included)
        # Calling function `init_fn(sess)` will load all the pretrained weights.
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

        # Initialization operation from scratch for the new "fc8" layers
        # `get_variables` will only return the variables whose name starts with the given pattern
        fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
        fc8_init = tf.variables_initializer(fc8_variables)

        # ---------------------------------------------------------------------
        # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
        # We can then call the total loss easily
        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.losses.get_total_loss()

        # First we want to train only the reinitialized last layer fc8 for a few epochs.
        # We run minimize the loss only with respect to the fc8 variables (weight and bias).
        fc8_optimizer = tf.train.AdamOptimizer(learning_rate= args.learning_rate1)
        fc8_train_op = fc8_optimizer.minimize(loss, var_list=fc8_variables)

        # Then we want to finetune the entire model for a few epochs.
        # We run minimize the loss only with respect to all the variables.
        full_optimizer = tf.train.AdamOptimizer(learning_rate= args.learning_rate2)
        full_train_op = full_optimizer.minimize(loss)

        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



        init_op = tf.global_variables_initializer()


        # tf.summary.image('images_'+ str(prediction[0]) , images, max_outputs=1)
        tf.summary.histogram("logits", logits)
        tf.summary.histogram("predictions", prediction)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("loss", loss)
        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

        tf.get_default_graph().finalize()
        

        
    # --------------------------------------------------------------------------
    # Now that we have built the graph and finalized it, we define the session.
    # The session is the interface to *run* the computational graph.
    # We can call our training operations with `sess.run(train_op)` for instance
    with tf.Session(graph=graph) as sess:
        print 'Input/default arguments: '
        pprint([(s,getattr(args,s)) for s in sorted(vars(args))])

        sess.run(init_op)
        init_fn(sess)  # load the pretrained weights
        sess.run(fc8_init)  # initialize the new fc8 layer
        logstring = args.logdir + '/' +  now.isoformat() + 'DROPOUT_' + str(args.dropout_keep_prob) + 'LR1_' + str(args.learning_rate1) + 'LR2_' + str(args.learning_rate2)
        train_writer = tf.summary.FileWriter(logstring +'_train' , graph=tf.get_default_graph())
        val_writer = tf.summary.FileWriter(logstring   + '_val', graph=tf.get_default_graph())
    
        batch_number = 1
        max_train_acc = args.save_if_train_acc_above
        max_valid_acc = args.save_if_valid_acc_above

        # Update only the last layer for a few epochs.
        for epoch in range(args.num_epochs1):
            print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
            
            val_accuracies = []
            train_accuracies = []
            


            sess.run(train_init_op)
            while True:
                try:
                    batch_number = batch_number + 1

                    train_accuracy, _, summary  = sess.run([accuracy, fc8_train_op, summary_op],  {is_training: True})
                    train_accuracies.append(train_accuracy)

                    train_writer.add_summary(summary, batch_number)


                except tf.errors.OutOfRangeError:
                    break

            
            sess.run(val_init_op)
            while True:
                try:
                    val_accuracy, summary = sess.run([accuracy,summary_op], {is_training: False})

                    val_accuracies.append(val_accuracy)

                except tf.errors.OutOfRangeError:
                    break
            val_writer.add_summary(summary, batch_number)
            
            batch_mean_train_acc = np.mean(train_accuracies)
            batch_mean_valid_acc = np.mean(val_accuracies)

            print '\ttrain - val accuracy: ' + np.array2string(batch_mean_train_acc) + ' - ' + np.array2string(batch_mean_valid_acc)
            if batch_mean_train_acc >= max_train_acc and batch_mean_valid_acc >= max_valid_acc:
                max_train_acc = batch_mean_train_acc
                max_valid_acc = batch_mean_valid_acc
                save_path = saver.save(sess, "tmp/new_model.ckpt")
                print("Model saved in path: %s" % save_path)


        # Train the entire model for a few more epochs, continuing with the *same* weights.
        for epoch in range(args.num_epochs1):
            print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
            
            val_accuracies = []
            train_accuracies = []
            
            

            sess.run(train_init_op)
            while True:
                try:
                    batch_number = batch_number + 1

                    train_accuracy, _, summary  = sess.run([accuracy, full_train_op, summary_op],  {is_training: True})
                    train_accuracies.append(train_accuracy)

                    train_writer.add_summary(summary, batch_number)
                except tf.errors.OutOfRangeError:
                    break

            sess.run(val_init_op)
            while True:
                try:
                    val_accuracy, summary = sess.run([accuracy,summary_op], {is_training: False})
                    val_accuracies.append(val_accuracy)

                except tf.errors.OutOfRangeError:
                    break
            val_writer.add_summary(summary, batch_number)

            batch_mean_train_acc = np.mean(train_accuracies)
            batch_mean_valid_acc = np.mean(val_accuracies)
            print '\ttrain - val accuracy: ' + np.array2string(np.mean(train_accuracies)) + ' - ' + np.array2string(np.mean(val_accuracies))
            if batch_mean_train_acc >= max_train_acc and batch_mean_valid_acc >= max_valid_acc:
                max_train_acc = batch_mean_train_acc
                max_valid_acc = batch_mean_valid_acc
                save_path = saver.save(sess, "tmp/new_model.ckpt")
                print("Model saved in path: %s" % save_path)
                    
save_path = saver.save(sess, "tmp/new_model_last_step.ckpt")
print("Last step model saved in path: %s" % save_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)