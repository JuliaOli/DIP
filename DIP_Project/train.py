import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import sys
import numpy as np

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Define a dict with the data-names and types we expect to
# find in the TFRecords file.
# Needs to be specified again,
# because it could have been written in the header of the
# TFRecords file instead.
def parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label":     tf.FixedLenFeature([], tf.int64)
    }
    # Parse the serialized data so we get a dict with our data
    parsed = tf.parse_single_example(record, keys_to_features)
    
    # Get the image as raw bytes.
    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    
    # The type is now uint8 but we need it to be float.
    image = tf.cast(image, tf.float32)

    # Get the label associated with the image.
    label = tf.cast(parsed["label"], tf.int32)

    return {'image': image}, label

# Helper-function for creating an input-function
# that reads from TFRecords files for use with the Estimator API.
def input_fn(filenames):
  
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=40)
    dataset = dataset.apply(
        tf.contrib.data.shuffle_and_repeat(1024, 1)
    )
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(parser, 32)
    )
    #dataset = dataset.map(parser, num_parallel_calls=12)
    #dataset = dataset.batch(batch_size=1000)
    dataset = dataset.prefetch(buffer_size=2)
    return dataset

#This is the input-function for the training-set for use with the Estimator API
def train_input_fn():
    return input_fn(filenames=["train.tfrecords", "test.tfrecords"])

#This is the input-function for the test-set for use with the Estimator API
def val_input_fn():
    return input_fn(filenames=["val.tfrecords"])

#   Create an arbitrary TensorFlow model.
#   Create a function which defines the following:
#       -The TensorFlow model, e.g. a Convolutional Neural Network.
#       -The output of the model.
#       -The loss-function used to improve the model during optimization.
#       -The optimization method.
#       -Performance metrics.
def model_fn(features, labels, mode, params):
    # Args:
    #
    # features: This is the x-arg from the input_fn.
    # labels:   This is the y-arg from the input_fn.
    # mode:     Either TRAIN, EVAL, or PREDICT
    # params:   User-defined hyper-parameters, e.g. learning-rate.
    
    num_classes = 3
    net = features["image"]

    net = tf.identity(net, name="input_tensor")
    
    net = tf.reshape(net, [-1, 224, 224, 3])    

    net = tf.identity(net, name="input_tensor_after")
    
    # Layer Convulutional 1
    net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                           filters=32, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    
    # Layer Convulutional 2
    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                           filters=64, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)  

    # Layer Convulutional 3
    net = tf.layers.conv2d(inputs=net, name='layer_conv3',
                           filters=64, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)    

    net = tf.contrib.layers.flatten(net)
    
    # First fully-connected / dense layer.
    # This uses the ReLU activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc1',
                        units=128, activation=tf.nn.relu)  
    
    net = tf.layers.dropout(net, rate=0.5, noise_shape=None, 
                        seed=None, training=(mode == tf.estimator.ModeKeys.TRAIN))
    
    # Logits output of the neural network.
    # Second fully-connected / dense layer.
    # This is the last layer so it does not use an activation function.
    logits = tf.layers.dense(inputs=net, name='layer_fc_2',
                        units=num_classes)

    # Softmax output of the neural network.
    y_pred = tf.nn.softmax(logits=logits)

    y_pred = tf.identity(y_pred, name="output_pred")

    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)

    y_pred_cls = tf.identity(y_pred_cls, name="output_cls")


    if mode == tf.estimator.ModeKeys.PREDICT:
        # If the estimator is supposed to be in prediction-mode
        # then use the predicted class-number that is output by
        # the neural network.
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:
        # Otherwise the estimator is supposed to be in either
        # training or evaluation-mode. Note that the loss-function
        # is also required in Evaluation mode.
        
        # Define the loss-function to be optimized, by first
        # calculating the cross-entropy between the output of
        # the neural network and the true labels for the input data.
        # This gives the cross-entropy for each image in the batch.
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)
        
        # Reduce the cross-entropy batch-tensor to a single number
        # which can be used in optimization of the neural network.
        loss = tf.reduce_mean(cross_entropy)

        # Define the optimizer for improving the neural network.
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        
        # Get the TensorFlow op for doing a single optimization step
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        
        # Define the evaluation metrics,
        # in this case the classification accuracy.
        metrics = {
            "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
        }

        # Wrap all of this in an EstimatorSpec
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
        
    return spec

#Create the estimator
model = tf.estimator.Estimator(model_fn=model_fn,
                               params={"learning_rate": 1e-4},
                               model_dir="./model5/")

count = 0

results_accuracy = []


while (count < 100):

    #Now that our new Estimator has been created, we can train it.
    model.train(input_fn=train_input_fn, steps=1000)

    # Once the model has been trained, 
    # we can evaluate its performance on the test-set.
    result = model.evaluate(input_fn=val_input_fn)

    print(result)
    print("Classification accuracy: {0:.2%}".format(result["accuracy"]))

    #Colleting the results in order to plot the graph of the accuracy    
    results_accuracy.append(result["accuracy"])

    sys.stdout.flush()
    count = count + 1

#Plotting the accuracy 
plt.plot(results_accuracy)
plt.title('Accuracy per Iteration')
plt.savefig('accuracy.jpg')