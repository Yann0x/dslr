import json
import sys

import lib
import pandas as pd


def predict(data: pd.DataFrame, model: lib.Model):
    pred = [0.0 for _ in range(len(model.houses_names))]
    print(f"Index,{model.house_field}")
    for student in data.index:
        scores = data.loc[student].values
        for idx, _ in enumerate(model.houses_names):
            pred[idx] = lib.dot(model.weights[idx], scores) + model.bias[idx]
        proba = lib.softmax(pred)
        res = model.houses_names[lib.argmax(proba)]
        print(f"{student},{res}")


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

    with open("model.json", "r") as file:
        dict_model = json.load(file)

    model = lib.Model(
        dict_model["weights"],
        dict_model["bias"],
        dict_model["houses_names"],
        dict_model["features"],
        dict_model["normalisation_method"],
    )

    data = df.filter(model.features)
    if model.normalisation_method == "standardisation":
        lib.standardisation(data)
    elif model.normalisation_method == "robust_scaling":
        lib.robust_scaling(data)
    else:
        sys.exit("No valid normalisation method in model specs")

    predict(data, model)


if __name__ == "__main__":
    main()
