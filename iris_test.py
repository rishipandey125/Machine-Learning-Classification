import matplotlib.pyplot as plt
import random_forest
import csv
"""
Testing Random forest Machine Learning Model with IRIS dataset.
Model is trained based off of 60% of classic iris data
and tested using the other 40% it has never seen.
Results comparing actual value and predicted value are graphed using matplotlib.

Data must follow the format of...
[feature1,feature2,feature3...featureN,Label]
to be interpreted.
A header must be included.
ie: ['sepal_length','sepeal_width',etc.,'species']
"""
with open('iris.csv') as csvfile:
    iris_data = list(csv.reader(csvfile, delimiter=','))

#Building an ML model with 128 trees and atleast x% accuracy.
iris_model = random_forest.buildForest(iris_data,.90,128)
#Differences between actual vals and predictions for testing data.
forest = iris_model.forest
differences = iris_model.differences
accuracy = iris_model.accuracy

iris_graph_key = {'setosa':1,'versicolor':2,'virginica':3}
actual = []
predicted_incorrectly = []
predicted_correctly = []
iterations = []
correct_iterations = []
incorrect_iterations = []

#Prepping Lists for Graphing
for y in range(1,len(differences)):
    iterations.append(y)
    actual.append(iris_graph_key[differences[y][0]])
    if differences[y][0] == differences[y][1]:
        correct_iterations.append(y)
        predicted_correctly.append(iris_graph_key[differences[y][1]])
    else:
        incorrect_iterations.append(y)
        predicted_incorrectly.append(iris_graph_key[differences[y][1]])

#Graphing Actual vs Predicted
fig, (act, pred) = plt.subplots(2,sharex=True,sharey=True)
plt.setp(act, xticks=iterations,xticklabels=[],yticks=[1, 2, 3],yticklabels=['Setosa','Versicolor','Virginica'])
fig.suptitle("Actual vs ML Predicted for IRIS Dataset")
act.scatter(iterations,actual,color='green')
act.title.set_text("Actual Species")
pred.scatter(correct_iterations,predicted_correctly,color='blue',label='Correct Prediction')
pred.scatter(incorrect_iterations,predicted_incorrectly,color='red',label='Incorrect Prediction')
fig.legend(loc=4)
pred.title.set_text("ML Predicted Species ({}% Accuracy)".format(str(round(accuracy*100,2))))
plt.show()
