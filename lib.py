import math
import sys

import matplotlib.pyplot as plt
import pandas as pd


class Model:
    def __init__(self, weights, bias, houses_names, features, normalisation_method):
        self.weights: list[list[float]] = weights
        self.bias: list[float] = bias
        self.house_field: str = "Hogwarts House"
        self.houses_names: list[str] = houses_names
        self.features: list[str] = features
        self.normalisation_method: str = normalisation_method


def mean(args: list) -> float | None:
    """Compute the arithmetic mean of the given values."""
    if not args:
        return None
    try:
        res = sum(elem for elem in args) / len(args)
    except Exception:
        return None
    return res


def median(args: list) -> float | int | None:
    """Compute the median of the given values."""
    if not args:
        return None
    try:
        args_list = sorted(args)
    except Exception:
        return None
    argssize = len(args_list) // 2
    if len(args_list) % 2 == 0:
        median = (args_list[argssize - 1] + args_list[argssize]) / 2
    else:
        median = args_list[argssize]
    return median


def quartile(args: list) -> list[float] | None:
    """Compute the first , second and third quartiles of the given values."""
    if not args:
        return None
    try:
        args_list = sorted(args)
    except Exception:
        return None
    argssize = len(args_list) - 1
    quarts = [argssize / 4, argssize / 2, argssize * 3 / 4]
    res = []
    for i in range(3):
        if quarts[i] != int(quarts[i]):
            index = int(quarts[i])
            rest = quarts[i] - int(quarts[i])
            res.append(args_list[index] * (1 - rest) + args_list[index + 1] * rest)
        else:
            res.append(args_list[int(quarts[i])])
    return res


def var(args: list) -> float | int | None:
    """Compute the variance of the given values."""
    if len(args) < 2:
        return None
    meaning = mean(args)
    if meaning is not None:
        variance = sum((x - meaning) ** 2 for x in args) / (len(args) - 1)
        return variance
    return None


def std(args: list) -> float | int | None:
    """Compute the standard deviation of the given values."""
    variance = var(args)
    if variance is not None:
        return variance**0.5
    return None


def min(args: list) -> float | int | None:
    """Returns the smallest value in the args"""
    if not args:
        return None
    min_val = args[0]
    for elem in args:
        if elem < min_val:
            min_val = elem
    return min_val


def max(args: list) -> float | int | None:
    """Returns the largest value in the args"""
    if not args:
        return None
    max_val = args[0]
    for elem in args:
        if elem > max_val:
            max_val = elem
    return max_val


def interquartil_range(values: list):
    """Compute the interquartil range (Q3 - Q1)"""
    quartiles = quartile(values)
    if not quartiles:
        return None
    q1 = quartiles[0]
    q3 = quartiles[2]
    return q3 - q1


def skewness(args: list) -> float | int | None:
    """Computes the skewness of the arg (is the list leaning left or right)"""
    args_mean = mean(args)
    args_std = std(args)
    if args_mean is not None and args_std is not None and args_std != 0:
        res = sum(((x - args_mean) ** 3) for x in args) / len(args) / args_std**3
    else:
        res = None
    return res


def standardisation(data):
    """Standardisation on data (using mean and std deviation)"""
    for field in data.columns:
        if field == "Hogwarts House":
            continue
        field_mean = mean(data[field].dropna().to_list())
        field_std = std(data[field].dropna().to_list())
        if not std:
            sys.exit("Error: standard deviation is 0")
        for i in data[field].index:
            if pd.isna(data.loc[i, field]):
                data.loc[i, field] = 0
                continue
            data.loc[i, field] = (data.loc[i, field] - field_mean) / field_std


def robust_scaling(data):
    """Robust Scaling (using median and interquartil range (Q3 - Q1))"""
    for field in data.columns:
        for field in data.columns:
            if field == "Hogwarts House":
                continue
        med = median(data[field].dropna().to_list())
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
    weights = model.weights
    bias = model.bias
    pred = [0.0 for _ in range(len(weights))]
    for idx, _ in enumerate(weights):
        pred[idx] = dot(weights[idx], scores) + bias[idx]
    return pred


def argmax(list: list[float]):
    max = 0
    for i, val in enumerate(list):
        if val > list[max]:
            max = i
    return max


def plot_accuracy(accuracy: list[float]):
    mean_acc: list[float] = [
        x if (x := mean(accuracy[:i])) is not None else 0.0
        for i in range(len(accuracy))
    ]
    plt.plot(mean_acc, label="Smoothed accuracy")
    plt.title("Training accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
