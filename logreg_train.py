import json
import math
import sys

import pandas as pd

import describe as yann


def interquartil_range(values: list):
    """Compute the interquartil range (Q3 - Q1)"""
    quartiles = yann.quartile(values)
    if not quartiles:
        return None
    q1 = quartiles[0]
    q3 = quartiles[2]
    return q3 - q1


def standardisation(data):
    """Standardisation on data (using mean and std deviation)"""
    for field in data.columns[1:]:
        mean = yann.mean(data[field].dropna().to_list())
        std = yann.my_std(data[field].dropna().to_list())
        if not std:
            sys.exit("Error: standard deviation is 0")
        for i in data[field].index:
            if pd.isna(data.loc[i, field]):
                data.loc[i, field] = mean
                continue
            data.loc[i, field] = (data.loc[i, field] - mean) / std


def robust_scaling(data):
    """Robust Scaling (using median and interquartil range (Q3 - Q1))"""
    for field in data.columns[1:]:
        med = yann.median(data[field].dropna().to_list())
        iqr = interquartil_range(data[field].dropna().to_list())
        if not iqr:
            sys.exit("Error: interquartil range is 0")
        for i in data[field].index:
            if pd.isna(data.loc[i, field]):
                data.loc[i, field] = 0
                continue
            data.loc[i, field] = (data.loc[i, field] - med) / iqr


def dot(v1: list, v2: list) -> float:
    if len(v1) != len(v2):
        print("Error: dot products require 2 vectors of the same length")
    res = 0
    for i1, i2 in zip(v1, v2):
        res += i1 * i2
    return res


def softmax(vals: list) -> list[float]:
    exp_vals = list(map(math.exp, vals))
    res = [0.0 for _ in range(len(vals))]
    sum_exp_vals = sum(exp_vals)
    for i in range(len(vals)):
        res[i] = exp_vals[i] / sum_exp_vals
    return res


def scale(vec: list[float], scl: float) -> list[float]:
    """Scale a vector by a scalar value."""
    scaled_vec = [x * scl for x in vec]
    return scaled_vec


def predict(model, scores) -> list[float]:
    weights = model[0]
    bias = model[1]
    pred = [0.0 for _ in range(len(weights))]
    for idx, _ in enumerate(weights):
        pred[idx] = dot(weights[idx], scores) + bias[idx]
    return pred


def train(data: pd.DataFrame, lr: float, epochs: int):

    to_predict = data.iloc[:, 0]
    houses = to_predict.unique()
    features = data.iloc[:, 1:]
    weights = [[1.0 for _ in range(4)] for _ in range(len(houses))]
    bias = [0.0 for _ in range(len(houses))]
    w_grad = [[0.0 for _ in range(4)] for _ in range(len(houses))]
    b_grad = [0.0 for _ in range(len(houses))]

    for epoch in range(epochs):
        for student in data.index:
            scores = features.loc[student].values

            pred = predict((weights, bias), scores)
            proba = softmax(pred)
            expected = list(houses).index(to_predict[student])
            loss = -math.log(proba[expected])
            print(f"for student{student} loss ->> {loss}")
            for idx, _ in enumerate(houses):
                truth = 0
                if idx == expected:
                    truth = 1
                grad = proba[idx] - truth
                w_grad[idx] = scale(scores, grad)
                b_grad[idx] = grad

            for idx, _ in enumerate(houses):
                for j in range(len(weights[idx])):
                    weights[idx][j] -= lr * w_grad[idx][j]
                bias[idx] -= lr * b_grad[idx]
    return (weights, bias)


def argmax(list: list[float]):
    max = 0
    for i, val in enumerate(list):
        if val > list[max]:
            max = i
    return max


def test_accuracy(data: pd.DataFrame, model: tuple[list[list[float]], list[float]]):
    to_predict = data.iloc[:, 0]
    houses = to_predict.unique()
    features = data.iloc[:, 1:]
    weights = model[0]
    bias = model[1]
    right = wrong = 0
    for student in data.index:
        scores = features.loc[student].values
        pred = predict((weights, bias), scores)
        proba = softmax(pred)
        result = argmax(proba)
        expected = list(houses).index(to_predict[student])
        if result == expected:
            right += 1
        else:
            wrong += 1

    acc = right / (right + wrong)
    print(f"{right} correct\n{wrong} incorrect\n {acc} accuracy")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <dataset.csv>")
        sys.exit(1)
    try:
        df = pd.read_csv(sys.argv[1])
    except FileNotFoundError:
        print(f"Error: file '{sys.argv[1]}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    fields_to_keep = [
        "Hogwarts House",
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Ancient Runes",
    ]
    data = df.filter(items=fields_to_keep)

    # standardisation(data)
    robust_scaling(data)

    model = train(data, lr=0.001, epochs=10)

    test_accuracy(data, model)
    with open("model.json", "w") as file:
        json.dump(model, file)


if __name__ == "__main__":
    main()
