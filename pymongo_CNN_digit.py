#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:31:52 2019

@author: changlinli
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils import shuffle
import os

Sample_rang = 2
imagesize = 28
Train_Steps = 2500
Train_epochs = 500
Batch_size = 500
Imagenum = 650

def incepter_process(datain,deeplevel,bnconfig):
    deepbas = 8
    layera = tf.contrib.layers.conv2d(datain,deepbas * 2 * deeplevel,1,normalizer_fn = tf.contrib.layers.batch_norm,normalizer_params = bnconfig)
    layerb = tf.contrib.layers.max_pool2d(layera, 2, stride = 2, padding = 'SAME')
    layerc = tf.contrib.layers.conv2d(layerb,deepbas*2*deeplevel*2,1,normalizer_fn = tf.contrib.layers.batch_norm,normalizer_params = bnconfig)
    layerd = tf.contrib.layers.max_pool2d(layerc, 2, stride = 2, padding = 'SAME')
    return layerd

def deep_main_model(features,labels,mode):
    bnsetting = {'is_training':mode==tf.estimator.ModeKeys.TRAIN,'decay':0.9,'zero_debias_moving_mean':False}
    input_layer = tf.reshape(features['pic'],[Batch_size,imagesize,imagesize,1])
    #print (input_layer.get_shape())
    Stage_a = incepter_process(input_layer,2,bnsetting)
    Stage_CONEND = tf.reshape(Stage_a, [Batch_size, -1])
    l = tf.shape(Stage_CONEND,name="SC_shape")
    Stage_Reduce = tf.contrib.layers.fully_connected(Stage_CONEND, 768,  normalizer_fn = tf.contrib.layers.batch_norm, normalizer_params = bnsetting)
    Logi = tf.contrib.layers.fully_connected(Stage_Reduce, 10, activation_fn = None)
    #print (Logi.get_shape())
    predictions = {
            "Pre" : tf.argmax(Logi, axis = 1),
            "Prob": tf.nn.softmax(Logi, name="Probout")
            }

#Predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = Logi)
    lr = tf.train.exponential_decay(0.01, tf.train.get_global_step(), 8000, 0.95)
    optimizer = tf.train.AdamOptimizer(learning_rate = lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

#Train
    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss = loss,
                    global_step = tf.train.get_global_step(),
                    aggregation_method = tf.AggregationMethod.EXPERIMENTAL_TREE)
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)
#Eval
    label = tf.argmax(labels, axis = 1)
    #print (label.get_shape())
    #print (predictions["Pre"].get_shape())
    eval_metric_ops = {
            "accuracy" : tf.metrics.accuracy(labels = label, predictions = predictions["Pre"])
            }
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)

def main(data,y,realtest):
    #shape : 42000*782 42000*28*28*1
    realtest = realtest.reshape((len(realtest),28,28,1))/255

    Xtrain = np.array(data[:int(len(data)*0.75),:],dtype = np.float32).reshape((int(len(data)*0.75),28,28,1))/255
    Xtest = np.array(data[int(len(data)*0.75):,:],dtype = np.float32).reshape((int(len(data)*0.25),28,28,1))/255
    ytrain =np.array(y[:int(len(data)*0.75)],dtype = np.int32).reshape((-1,1))
    ytest = np.array(y[int(len(data)*0.75):],dtype = np.int32).reshape((-1,1))
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"pic" : Xtrain},
            y = ytrain,
            batch_size = Batch_size,
            num_epochs = Train_epochs,
            shuffle = True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"pic" : Xtest},
            y = ytest,
            batch_size = Batch_size,
            num_epochs = 1,
            shuffle = False)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"pic" : realtest},
        batch_size = Batch_size,
        num_epochs =1,
        shuffle = False)
    tensorlog = {"Prob" : "Probout","shape":"SC_shape"}
    dnnc_reg_fn = deep_main_model
    log_hook = tf.train.LoggingTensorHook(tensors = tensorlog, every_n_iter = 5)
    train_spec = tf.estimator.TrainSpec(input_fn = train_input_fn, max_steps = Train_Steps, hooks = [log_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn = eval_input_fn, throttle_secs=240,hooks=[log_hook])
    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    estconfig = tf.estimator.RunConfig(session_config = sessconfig, save_summary_steps = 50, log_step_count_steps = 50)
    dnn_model = tf.estimator.Estimator(model_fn = dnnc_reg_fn, model_dir = "./DNN_Weight", config = estconfig)
    tf.estimator.train_and_evaluate(dnn_model, train_spec, eval_spec)
    pred_results = dnn_model.predict(predict_input_fn)
    get_answer = list(pred_results)
    return get_answer

if __name__ == "__main__":
     res = main(x_train,y,x_test)

ImageId = []
Label = []
for i in range(len(res)):
    ImageId.append(i+1)
    Label.append(res[i]['Pre'])
sub = pd.DataFrame({'ImageId':ImageId,'Label':Label})
sub.to_csv('submission.csv',index=False)
