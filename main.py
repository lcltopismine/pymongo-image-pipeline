#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 20:16:07 2019

@author: changlinli
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymongo
import gridfs
from tqdm import tqdm
from bson import objectid
import base64
import os
import scipy.misc
import matplotlib.image as mpimg

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

image_shape = 28
train_size,test_size = len(train),len(test)
print("The size of train data is: ,", train.shape)
print("The size of test data is: ,", test.shape)

label = pd.DataFrame()
label["id"] = pd.to_numeric([i for i in range(len(train))], downcast='integer')
label["label"] = pd.to_numeric(train["label"], downcast='integer')
train.drop(["label"],axis=1,inplace=True)

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
##create database,vaild til contents inserted
mydb = myclient["imagedatas"]
print(myclient.list_database_names())

mycol = mydb["label"]
##Insert all of the labels
label_list = label.to_dict("record")
mycol.insert_many(label_list)

for x in mycol.find({},{"_id":0}):
    print(x)
    
##load data into mongodb
train = train.values.reshape((train_size,image_shape,image_shape))
test = test.values.reshape((test_size,image_shape,image_shape))
fs = gridfs.GridFS(mydb)
if not os.path.exists('./trainimg'):
    os.mkdir("./trainimg")
if not os.path.exists('./testimg'):
    os.mkdir("./testimg")
    

for i in tqdm(range(train.shape[0]),total=train.shape[0]):
    scipy.misc.toimage(train[i,:], cmin=0.0).save('./trainimg/file_'+str(i)+'.jpg')
    with open('./trainimg/file_'+str(i)+'.jpg','rb') as file:
        string = base64.b64encode(file.read())
        upload = fs.put(string,content_type="Trainimage_"+str(i),filename="train_file_"+str(i)+".jpg",label=int(label["label"][i]))
        
for i in tqdm(range(test.shape[0]),total=test.shape[0]):
    scipy.misc.toimage(test[i,:], cmin=0.0).save('./testimg/file_'+str(i)+'.jpg')
    with open('./testimg/file_'+str(i)+'.jpg','rb') as file:
        string = base64.b64encode(file.read())
        upload = fs.put(string,content_type="Testimage_"+str(i),filename="test_file_"+str(i)+".jpg")
    
#Reference: https://psabhay.com/2015/03/mongodb-gridfs-using-python/
#Extract train image, train label, test image
image = mydb["fs.files"]
if not os.path.exists('./data'):
    os.mkdir('./data')
    os.mkdir("./data/trainimg")
    os.mkdir("./data/testimg")
    
query = { "filename": { "$regex": "^train" } }
num = 0
y = []
x_train, x_test = np.zeros((train_size,image_shape,image_shape)),np.zeros((test_size,image_shape,image_shape))
for x in tqdm(image.find(query),total=train_size):
    imgtrain = fs.get(x["_id"]).read()
    fh = open("./data/trainimg/trainimg_"+str(num)+".jpg", "wb")
    fh.write(base64.b64decode(imgtrain))
    fh.close()
    y.append(x["label"])
    #TO DO: ADD RE_SIZE COMMAND for size not fit
    imgtrain = mpimg.imread("./data/trainimg/trainimg_"+str(num)+".jpg")
    x_train[num,:] = imgtrain
    num+=1

query = { "filename": { "$regex": "^test" } }
num = 0
for x in tqdm(image.find(query),total=test_size):
    imgtest = fs.get(x["_id"]).read()
    fh = open("./data/testimg/testimg_"+str(num)+".jpg", "wb")
    fh.write(base64.b64decode(imgtest))
    fh.close()
    imgtest = mpimg.imread("./data/testimg/testimg_"+str(num)+".jpg")
    x_test[num,:] = imgtest
    num+=1
