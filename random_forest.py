import decision_tree
import random
"""
Random forest Machine Learning Model by Rishi Pandey.
Generates a Random forest Machine Learning Model.
Model is trained based off of data loaded as CSV.
"""

"""
Object for a Random Forest ML Model.
Contains a forest of decision trees, accuracy of model, and differences between actual and predicted.
"""
class RFModel:
    def __init__(self, forest, accuracy, differences):
        self.forest = forest
        self.accuracy = accuracy
        self.differences = differences

"""
Returns a bagged dataset to generate new decision tree.
Bagged dataset must have the same length as the original, but number of features will be randomized.
Bagged dataset has the same header as the rest of the data.
"""
def bagged(data):
    rand_num_features = random.randint(1,len(data[0])-1)
    feature_indices = []
    while rand_num_features != 0:
        random_feature_index = random.randint(0,len(data[0])-2)
        if random_feature_index not in feature_indices:
            feature_indices.append(random_feature_index)
        else:
            continue
        rand_num_features -= 1
    feature_indices.append(len(data[0])-1)
    bagged_data = [[data[0][i] for i in feature_indices]]
    #Starting at y = 1 as first entry is the header.
    for y in range(1,len(data)):
        index = random.randint(1,len(data)-1)
        bagged_data.append([data[index][i] for i in feature_indices])
    return bagged_data


"""
Build Random Forest Machine Learning Model based on data passed.
Model is created on 60% of the data passed and tested using the other 40% of data it has never seen.
acceptedAccuracy parameter specifies the accuracy the model should support.
numTrees paramter specifies the depth of the forest.
If the model is created with less accuracy than the specificed ammount, it will be scrapped and recreated.
"""

def buildForest(original_data,acceptedAccuracy,numTrees):
    differences = [['Actual','Predicted']]
    data = sorted(original_data[1:], key = lambda x: random.random())
    data.insert(0,original_data[0])
    training_data = data[:int(.6*len(data))]
    testing_data = data[int(.6*len(data)):]
    testing_data.insert(0,data[0])
    forest = [decision_tree.buildTree(training_data)]
    for x in range(numTrees-2):
        forest.append(decision_tree.buildTree(bagged(training_data)))
    """
    Tests a Random forest Machine Learning Model for Accuracy.
    Model and Testing Data are passed as parameters.
    Model is created on 60% of the data passed and tested using the other 40% of data it has never seen.
    Returns a float of the accuracy (ie: 0.91).
    """
    def model_accuracy(forest,testing_data):
        count = 0
        #loop through testing_data
        for y in range(1,len(testing_data)):
            input = {}
            #loop through features
            for x in range(len(testing_data[0])-1):
                input.update({testing_data[0][x]:testing_data[y][x]})
            #expected label
            actual = testing_data[y][len(testing_data[y])-1]
            predictions = []
            # loop through trees in forest
            for tree in forest:
                predictions.append(decision_tree.traverseTree(input,tree.node))
            distinct_prediction = {}
            # loop through and aggregate predictions
            for x in predictions:
                if not (isinstance(x,dict)):
                    if x not in distinct_prediction:
                        distinct_prediction.update({x:predictions.count(x)})
                    else:
                      continue
                else:
                    for i in x:
                        try:
                            val = distinct_prediction[i] + x[i]
                            distinct_prediction.pop(i,None)
                            distinct_prediction.update({i:val})
                        except KeyError:
                            continue
            predicted = max(distinct_prediction, key=distinct_prediction.get)
            differences.append([actual,predicted])
            #count accurate prediction
            if actual == predicted:
                count += 1
        return float(count)/float((len(testing_data)-1))

    accuracy = model_accuracy(forest,testing_data)
    if accuracy < acceptedAccuracy:
        print("Accuracy Test Failed "+ str(round(accuracy*100,2))+"%" + " Rebuilding...")
        return buildForest(data,acceptedAccuracy,numTrees)
    else:
        print("Passed! Model Built w/ " + str(round(accuracy*100,2)) + "%")
        forest.append(decision_tree.buildTree(data))
        return RFModel(forest,accuracy,differences)
