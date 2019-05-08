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
mydb = myclient["imagedata"]
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
import scipy.misc
for i in tqdm(range(train.shape[0]),total=train.shape[0]):
    scipy.misc.toimage(train[i,:], cmin=0.0).save('./trainimg/file_'+str(i)+'.jpg')
    with open('./trainimg/file_'+str(i)+'.jpg','rb') as file:
        string = base64.b64encode(file.read())
        upload = fs.put(string,content_type="Trainimage_"+str(i),filename="train_file_"+str(i)+".jpg")
        
for i in tqdm(range(test.shape[0]),total=test.shape[0]):
    scipy.misc.toimage(test[i,:], cmin=0.0).save('./testimg/file_'+str(i)+'.jpg')
    with open('./testimg/file_'+str(i)+'.jpg','rb') as file:
        string = base64.b64encode(file.read())
        upload = fs.put(string,content_type="Testimage_"+str(i),filename="test_file_"+str(i)+".jpg")
    
#Reference: https://psabhay.com/2015/03/mongodb-gridfs-using-python/
image = mydb["fs.files"]
for x in image.find({},{"_id":1}).limit(1):
    print(x)
imgtest = fs.get(x["_id"]).read()

fh = open("./test2.jpg", "wb")
fh.write(base64.b64decode(imgtest))
fh.close()



    