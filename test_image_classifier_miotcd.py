from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import math
import csv
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
slim = tf.contrib.slim
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_string(
    'test_dir', None, 'Test image directory.')
tf.app.flags.DEFINE_string('results_dir','','The directory where the prediction results are written')
tf.app.flags.DEFINE_string('label_path','datasets/labels.txt','The path to the labels.txt')
tf.app.flags.DEFINE_integer(
    'num_classes', 11, 'Number of classes.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'test_image_size', None, 'Eval image size')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

FLAGS = tf.app.flags.FLAGS
def main(_):
#    if not FLAGS.test_list:
#        raise ValueError('You must supply the test list with --test_list')

    if not FLAGS.test_dir:
        raise  ValueError('You must supply the test image directory')
    image_dir = FLAGS.test_dir
    tf.logging.set_verbosity(tf.logging.INFO)

    num_images = len(os.listdir(image_dir))
    file_names = os.listdir(image_dir)

    file_full_names=[]
    for image_id in range(num_images):
        file_full_name = os.path.join(image_dir,file_names[image_id])
        file_full_names.append(file_full_name)


    labels_fd=open(FLAGS.label_path,"r")
    class_names = [row.strip().split(':')[1] for row in labels_fd]
    print(class_names)

    csv_results_file_fullname =os.path.join(FLAGS.results_dir, FLAGS.model_name+"_results_test.csv")
    print(FLAGS.results_dir)
    print(csv_results_file_fullname)

    csv_out = open(csv_results_file_fullname,'wb')
    results_csv = csv.writer(csv_out,delimiter='\t')

    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=FLAGS.num_classes,
            is_training=False)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        test_image_size = FLAGS.test_image_size or network_fn.default_image_size

        image_string = tf.placeholder(tf.string)
        image = tf.image.decode_jpeg(image_string, channels=3,
                                     try_recover_truncated=True,
                                     acceptable_fraction=0.3)

        processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)
        processed_image = tf.expand_dims(processed_image, 0)

        ####################
        # Define the model #
        ####################
        logits, _ = network_fn(processed_image)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)

        with tf.Session() as sess:
            init_fn(sess)
            for file_full_name in file_full_names:
                x = open(file_full_name).read()
                pred = sess.run(predictions, feed_dict={image_string: x})
                print("the predict result of {}.jpg is {}".format(file_full_name.split('/')[-1].split('.')[0],class_names[int(pred[0])]))
                results_csv.writerow([file_full_name.split('/')[-1].split('.')[0], class_names[int(pred[0])]])

    csv_out.close()


if __name__ == '__main__':
    tf.app.run()
