import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn import svm


def SkewDisposal(data):
    data["total_rooms"] = np.log(data["total_rooms"] + 1)
    data["total_bedrooms"] = np.log(data["total_bedrooms"] + 1)
    data["population"] = np.log(data["population"] + 1)
    data["households"] = np.log(data["households"] + 1)


def Preprocessing(size=0.2):
    data = pd.read_csv("housing.csv")
    print("Mean value", data["median_house_value"].mean())
    data.dropna(inplace=True)

    SkewDisposal(data)
    data = data.join(pd.get_dummies(data.ocean_proximity)).drop(["ocean_proximity"], axis=1)
    data["bedroom_ratio"] = data["total_bedrooms"] / data["total_rooms"]

    trainData, testData = train_test_split(data, test_size=size)
    xTrain, yTrain = trainData.drop(["median_house_value"], axis=1), trainData["median_house_value"]
    xTest, yTest = testData.drop(["median_house_value"], axis=1), testData["median_house_value"]
    return xTrain, yTrain, xTest, yTest


def Graphics(result, yTest):
    plt.plot(yTest, abs(result - yTest) / yTest)
    plt.show()


def GraphicRMSE(resultLR, resultRF):
    for i in range(3, 7):
        xTrain, yTrain, xTest, yTest = Preprocessing(i*0.1)
        resultLR.append(LinRegression(xTrain, yTrain, xTest, yTest))
        resultRF.append(ForestReg(xTrain, yTrain, xTest, yTest))
    plt.plot([0.2, 0.3, 0.4, 0.5, 0.6], resultLR)
    plt.plot([0.2, 0.3, 0.4, 0.5, 0.6], resultRF)
    plt.xlabel("size for train")
    plt.ylabel("RMSE")
    plt.legend(["LR", "RF"])
    plt.show()


def ComputeErrors(str, yPred, yTest):
    print("\n" + str + "\n")
    print("Mean error in %:", 100 * mean_absolute_percentage_error(yTest, yPred))
    print("MAE:", mean_absolute_error(yTest, yPred))
    print("MSE:", mean_squared_error(yTest, yPred))
    print("RMSE:", root_mean_squared_error(yTest, yPred))


def LinRegression(xTrain, yTrain, xTest, yTest):
    reg = LinearRegression()
    reg.fit(xTrain, yTrain)
    result = reg.predict(xTest)
    # Graphics(result, yTest)
    # ComputeErrors("Linear Regression", result, yTest)
    return result


def SVM(xTrain, yTrain, xTest, yTest):
    clf = svm.SVC()
    clf.fit(xTrain, yTrain)
    result = clf.predict(xTest)
    Graphics(result, yTest)
    ComputeErrors("SVM Algorithm", result, yTest)


def ForestReg(xTrain, yTrain, xTest, yTest):
    forest = RandomForestRegressor()
    forest.fit(xTrain, yTrain)
    result = forest.predict(xTest)
    # Graphics(result, yTest)
    # ComputeErrors("Random Forest Algorithm", result, yTest)
    return result


if __name__ == '__main__':
    xTrain, yTrain, xTest, yTest = Preprocessing()
    lr =LinRegression(xTrain, yTrain, xTest, yTest)
    rf = ForestReg(xTrain, yTrain, xTest, yTest)
    print("Actual house value:", yTest.values[0])
    print("Predicted by LR house value", lr[0])
    print("Predicted by RF house value", rf[0], "\n")
    print("Actual house value:", yTest.values[1])
    print("Predicted by LR house value", lr[1])
    print("Predicted by RF house value", rf[1])
    # GraphicRMSE(resultLR, resultRF)


    # SVM(xTrain, yTrain, xTest, yTest)


