from imageai.Prediction import ImagePrediction
from os import getcwd,path
execution_path=getcwd()
#test

prediction = ImagePrediction()
prediction.setModelTypeAsDenseNet()
prediction.setModelPath(path.join(execution_path, "DenseNet-BC-121-32.h5"))
prediction.loadModel()

predictions, probabilities = prediction.predictImage(path.join(execution_path, "giraffe.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
