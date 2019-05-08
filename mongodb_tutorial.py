#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:57:26 2019

@author: changlinli
"""

import pymongo
import gridfs

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
##create database,vaild til contents inserted
mydb = myclient["imagedata"]
print(myclient.list_database_names())
##create collection,vaild til contents inserted
mycol = mydb["customers"]
print(mydb.list_collection_names())

##Insert-one operation
mydict = { "name": "John", "address": "Highway 37" }
x = mycol.insert_one(mydict)
print(x.inserted_id)

##Insert-multiple operations
mylist = [
  { "name": "Amy", "address": "Apple st 652","age":1},
  { "name": "Hannah", "address": "Mountain 21"},
  { "name": "Michael", "address": "Valley 345"},
  { "name": "Sandy", "address": "Ocean blvd 2"},
  { "name": "Betty", "address": "Green Grass 1"},
  { "name": "Richard", "address": "Sky st 331"},
  { "name": "Susan", "address": "One way 98"},
  { "name": "Vicky", "address": "Yellow Garden 2"},
  { "name": "Ben", "address": "Park Lane 38"},
  { "name": "William", "address": "Central st 954"},
  { "name": "Chuck", "address": "Main Road 989"},
  { "name": "Viola", "address": "Sideway 1633"}##A list of dictionary
]

x = mycol.insert_many(mylist)

#print list of the _id values of the inserted documents:
print(x.inserted_ids)

### Select * --- Find
#para>>{"_id":0,"name":1,"address":1}
for x in mycol.find({},{"_id":0}):
    ##dict operations
    print(x)

##Limit your output
for x in mycol.find({},{"_id":0}).limit(3):
    ##dict operations
    print(x)
    
###Where statement
where = { "address": { "$gt": "S" } }
for x in mycol.find(where,{"_id":0}):
    print(x)

###sort the result by key, second argument: 1>>up -1>>down
where = { "address": { "$gt": "S" } }
for x in mycol.find(where,{"_id":0}).sort("name",-1):
    print(x)
    
###delete operation, delete one
myquery = { "address": "Mountain 21" }
x=mycol.delete_one(myquery)

where = { "address": { "$gt": "S" } }
mycol.delete_many(where)

###drop current table
mycol.drop()

##Update operation
myquery = { "address": "Valley 345" }
newvalues = { "$set": { "address": "Canyon 123" } }

mycol.update_one(myquery, newvalues)

#print "customers" after the update:
for x in mycol.find():
  print(x)
  
myquery = { "address": { "$regex": "^S" } }
newvalues = { "$set": { "name": "Minnie" } }

x = mycol.update_many(myquery, newvalues)

print(x.modified_count, "documents updated.")

