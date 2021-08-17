#Here this whole project is a modular programming approach.Here i have build a multiple text analyzer inside
#one project ,for example--if i have a review of some product of amazon such as headphone then i can classify these review
#But if i want to analyze the review of any other product or any other headphone then i have to build another model for this
#  but by this approach which i have applied here i can make do this work in a single project  ,Means here i can take many stopwords
# and then do predictions by creating directery for all stopwords,consider if i have one type of  headphone then i can create a
# directry of headphone and create another directry inside this which defines the type 1 or or 3.. and preprocess stopwords only for
# this directry and train and predict for this ,same if i have multiple organizations and there are many departments(like support)
# inside these organizations  if i have to analyze review for these all departments then i can make directry and get stopwords and train


#Here we can use postman to send request for create training stopwords and prediction ,by giving userid(product name)and projectid(producxt type)

from wsgiref import simple_server
from flask import Flask, request
from flask import Response
import os
from flask_cors import CORS,cross_origin  ## CORS(Cross-origin share resourse) is use to acces the request from other domain
# (means which have another host or port or domain name ),here we are also gona use postman to make request

import json

from prediction.predictApp import PredictApi
from training.trainApp import TrainApi
from preprocessing.preprocessing import createDirectoryForUser, extractDataFromTrainingIntoDictionary, \
    deleteExistingTrainingFolder

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)
# app.config['DEBUG'] = True

trainingDataFolderPath = "trainingData/"


class StartApi:
    def __init__(self):
        stopWordsFilePath = "data/stopwords.txt"
        self.predictObj = PredictApi(stopWordsFilePath)#Making object of PredictApi class which is created inside predictApp.py
        self.trainObj = TrainApi(stopWordsFilePath)#Making object of TrainApi class which is created inside trainApp.py


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        if request.json['text'] is not None and request.json['userId'] is not None and request.json['projectId'] is not None:
            text = request.json['text']
            userId = str(request.json['userId'])
            projectId = str(request.json['projectId'])
            #csvFilePath = trainingData + userId + "/" + projectId + "/trainingData.csv"
            jsonFilePath = trainingDataFolderPath + userId + "/" + projectId + "/trainingData.json"
            modelPath = trainingDataFolderPath + userId + "/" + projectId + "/modelForPrediction.sav"
            vectorPath = trainingDataFolderPath + userId + "/" + projectId + "/vectorizer.pickle"
            result = StartApp.predictObj.executePreocessing(text, jsonFilePath,modelPath,vectorPath)


    except ValueError:
        return Response("Value not found inside  json trainingData")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        return Response((str(e)))
    return Response(result)


@app.route("/train", methods=['POST'])
@cross_origin()
def trainModel():

    try:
        if request.get_json() is not None: # check the json data given(through post or by another way)is None or not
            data = request.json['data'] # Take data(json data given through postman ) inside the data variable
        if request.json['userId'] is not None:
            userId = str(request.json['userId'])# Getting user id
            # path = trainingData+userId
        if request.json['projectId'] is not None:
            projectId = str(request.json['projectId'])# Getting projectid
            # path = path + "/" + projectId
            #path = "C:\\Users\\user\\PycharmProjects\\TwitterSentimentAnalysis\\Twitter.json"

            createDirectoryForUser(userId, projectId)# calling a function which creates directry,above i have import this
            #  function from preprocessing

        path = trainingDataFolderPath + userId + "/" + projectId

        trainingDataDict = extractDataFromTrainingIntoDictionary(data)# calling function to extract data or allign data in a
        # manner such that data allign in key value pair

        with open(path + '/trainingData.json', 'w', encoding='utf-8') as f:# here we are opening data file and give file access
            # to f object .UTF-8 is an encoding system for Unicode. It can translate
            # any Unicode character to a matching unique binary string, and can also translate the binary string back to
            #  Unicode character. This is the meaning of “UTF”, or “Unicode Transformation Format.”
            json.dump(trainingDataDict, f, ensure_ascii=False, indent=4)
        #dataFrame = pd.read_json(path + '/trainingData.json')
        jsonpath = path + '/trainingData.json'
        modelPath = path
        modelscore = StartApp.trainObj.training_model(jsonpath,modelPath)
        #dataFrame.to_csv(path + '/trainingData.csv', index=None, header=True)
    except ValueError as val:
        return Response("Value not found inside  json trainingData", val)
    except KeyError as keyval:
        return Response("Key value error incorrect key passed", keyval)
    except Exception as e:
        return Response((str(e)))

    return Response("Success")


@app.route("/deleteuserproject", methods=["GET"])
@cross_origin()
def deleteUserProjectFolder():
    try:
        if request.args.get("userId") is not None:
            userIdAndProjectId = trainingDataFolderPath + request.args.get("userId")
        if request.args.get("projectId") is not None:
            userIdAndProjectId = userIdAndProjectId + "/" + request.args.get("projectId")

        # pathForExitingFolder = "ids/" + userId
        if userIdAndProjectId is not None:
            deleteExistingTrainingFolder(userIdAndProjectId)
        else:
            return Response("Please check your input")

    except Exception as e:
        return Response("Please check your input", e)
    return "Operation Successfully completed"


@app.route("/noofusers", methods=["GET"])
@cross_origin()
def getTrainingImagesFolders():
    try:
        for root, dirs, files in os.walk("trainingData"):
            print("%s these are the images you have trained so far" % dirs)
            return Response("%s these are the images you have trained so far" % dirs)
        return "We don't have any images for training so far"
    except Exception as e:
        return e
    return "We don't have any images for training so far"


#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    StartApp = StartApi()
    app.run(port=8080,debug=True)

