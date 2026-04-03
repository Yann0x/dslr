import math

import pandas as pd

import describe as yann


class Model:
    def __init__(self, weights, bias, house_field, houses_names, features):
        self.weights: list[list[float]] = weights
        self.bias: list[float] = bias
        self.house_field: str = house_field
        self.houses_names: list[str] = houses_names
        self.features: list[str] = features


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
                data.loc[i, field] = 0
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
