import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import pandas as pd


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
ALGORITHM = "custom_net"
# ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return x * (1 - x)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        if minibatches:
            for i in range(0, epochs):
                xBatches, yBatches = self.__batchGenerator(xVals, mbs), self.__batchGenerator(yVals, mbs)
                loss = 0.0
                for xBatch, yBatch in zip(xBatches, yBatches):
                    L1Out, L2Out = self.__forward(xBatch)
                    loss_batch = self.__loss(yBatch, L2Out)
                    loss = loss + loss_batch
                    L2E = L2Out - yBatch
                    L2D = L2E * self.__sigmoidDerivative(L2Out)
                    L1E = np.matmul(L2D, self.W2.T)
                    L1D = L1E * (self.__sigmoidDerivative(L1Out))
                    L1A = np.matmul(xBatch.T, L1D) * self.lr
                    L2A = np.matmul(L1Out.T, L2D) * self.lr
                    W1New = self.W1 - L1A
                    W2New = self.W2 - L2A
                    self.W1 = W1New
                    self.W2 = W2New
                loss = loss / mbs
                print('Training epoch %d, Loss: %f' % (i, loss))
        else:
            for i in range(0, epochs):
                L1Out, L2Out = self.__forward(xVals)
                loss = self.__loss(yVals, L2Out)
                print('Training epoch %d, Loss: %f' % (i, loss))
                L2E = L2Out - yVals
                L2D = L2E * self.__sigmoidDerivative(L2Out)
                L1E = np.matmul(L2D, self.W2.T)
                L1D = L1E * (self.__sigmoidDerivative(L1Out))
                L1A = np.matmul(xVals.T, L1D) * self.lr
                L2A = np.matmul(L1Out.T, L2D) * self.lr
                W1New = self.W1 - L1A
                W2New = self.W2 - L2A
                self.W1 = W1New
                self.W2 = W2New

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        pred = np.zeros_like(layer2)
        pred[np.arange(len(layer2)), layer2.argmax(1)] = 1
        return pred

        # Forward pass.

    def __loss(self, label, pred):
        mse = np.average(np.sum(np.square(label - pred), axis=1))
        return mse


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))

def getRawDataIris():
    iris_path = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(iris_path, sep=',')
    return df


def preprocessDataIris(raw):
    iris = raw.to_numpy()
    np.random.shuffle(iris)
    print(iris.shape)
    print(np.unique(iris[:, 4]))
    for i in range(iris.shape[0]):
        if iris[i, 4] == 'Iris-setosa':
            iris[i, 4] = 0
        elif iris[i, 4] == 'Iris-versicolor':
            iris[i, 4] = 1
        elif iris[i, 4] == 'Iris-virginica':
            iris[i, 4] = 2
    trainSize = int(iris.shape[0] * 0.8)
    xTrain = iris[:trainSize, 0:3].astype(np.float)
    yTrain = iris[:trainSize, 4]
    xTest = iris[trainSize:, 0:3].astype(np.float)
    yTest = iris[trainSize:, 4]
    yTrain = to_categorical(yTrain, 3).astype(int)
    yTest = to_categorical(yTest, 3).astype(int)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain, xTest = np.reshape(xTrain, [-1, IMAGE_SIZE]), np.reshape(xTest, [-1, IMAGE_SIZE])
    xTrain, xTest = xTrain / 255.0, xTest / 255.0
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        custom_net = NeuralNetwork_2Layer(xTrain.shape[1], yTrain.shape[1], neuronsPerLayer=20)
        custom_net.train(xTrain, yTrain, epochs=20, minibatches=True, mbs=32)#TODO: Write code to build and train your custon neural net.
        return custom_net
    elif ALGORITHM == "tf_net":
        yTrainClass = np.argmax(yTrain, axis=1)
        model = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(128, activation=tf.nn.sigmoid), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(xTrain, yTrainClass, epochs=25)
        return model
    else:
        raise ValueError("Algorithm not recognized.")


def trainModelIris(data):
    xTrain, yTrain = data
    custom_net = NeuralNetwork_2Layer(xTrain.shape[1], yTrain.shape[1], neuronsPerLayer=50)
    custom_net.train(xTrain, yTrain, epochs=50, minibatches=True, mbs=32)   #TODO: Write code to build and train your custon neural net.
    return custom_net

def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        out = model.predict(data)
        pred = np.zeros_like(out)
        pred[np.arange(len(out)), out.argmax(1)] = 1
        return pred
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    confuse_table = np.zeros([preds.shape[1], preds.shape[1]])
    f1_list = np.zeros(preds.shape[1])
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):
            acc = acc + 1
        y_label_class = np.argmax(yTest[i])
        y_pred_class = np.argmax(preds[i])
        confuse_table[y_label_class][y_pred_class] = confuse_table[y_label_class][y_pred_class] + 1
    confuse_table = confuse_table.astype(int)

    for i in range(confuse_table.shape[0]): # iterate over each true class
        tp = confuse_table[i][i]
        fn = np.sum(confuse_table[i]) - tp     # sum over row
        fp = np.sum(confuse_table[:][i]) - tp   # sum over column
        if (tp + 0.5*(fp + fn)) == 0:
            f1_list[i] = 0.0
        else:
            f1_list[i] = tp / (tp + 0.5*(fp + fn))

    accuracy = acc / preds.shape[0]

    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print("Confusion Matrix (True Class \  Pred Class) :\n%s" % str(confuse_table))
    print("F1 for each class: \n%s" % str(f1_list))
    print("Macro-Averaged F1: %s" % str(np.average(f1_list)))



#=========================<Main>================================================

def main():
    print("Classification on MNIST")
    # classification on MNIST
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)

    # classification on IRIS
    print("Classification on IRIS")
    raw = getRawDataIris()
    data = preprocessDataIris(raw)
    model = trainModelIris(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)




if __name__ == '__main__':
    main()
