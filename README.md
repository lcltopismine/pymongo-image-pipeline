# pymongo-image-pipeline
##### This is a pipeline to save picture in and export them by introducing mongodb(in python).
##### Quickly go through following,
###### 0. data acknowledgement: from kaggle platform, digits recognizer
###### 1. Load in data, insert/delete/update entries
###### 2. convert image to bytes and load them to GridFs,which is a MongoDB specification for storing and retrieving large files such as images, audio files, video files, etc.
###### 3. Re-cast back to normal image and save them locally. It is possible to go through all of the pictures but I don't have so much hard drive space at this stage -_-
###### 4. Automate import/export process, and a CNN model to check if export image works.
###### 5. #Tell me if there are some TODOs,thx.

