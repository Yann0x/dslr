import json
import sys

import pandas as pd

import lib as lib


def train(data: pd.DataFrame, lr: float, epochs: int):

    to_predict = data.iloc[:, 0]
    houses = data.iloc[:, 0].unique()
    features = data.iloc[:, 1:]
    model = lib.Model(
        weights=[
            [1.0 for _ in range(len(features.columns))] for _ in range(len(houses))
        ],
        bias=[0.0 for _ in range(len(houses))],
        house_field=data.columns[0],
        houses_names=houses.tolist(),
        features=data.iloc[:, 1:].columns.to_list(),
    )
    w_grad = [
        [0.0 for _ in range(len(model.weights[0]))] for _ in range(len(model.weights))
    ]
    b_grad = [0.0 for _ in range(len(model.bias))]
    for epoch in range(epochs):
        for student in data.index:
            scores = features.loc[student].values
            pred = lib.predict(model, scores)
            proba = lib.softmax(pred)
            expected = list(houses).index(to_predict[student])
            # loss = -math.log(proba[expected])
            # print(f"for student{student} loss ->> {loss}")
            for idx, _ in enumerate(houses):
                truth = 0
                if idx == expected:
                    truth = 1
                grad = proba[idx] - truth
                w_grad[idx] = lib.scale(scores, grad)
                b_grad[idx] = grad

            for idx, _ in enumerate(houses):
                for j in range(len(model.weights[idx])):
                    model.weights[idx][j] -= lr * w_grad[idx][j]
                model.bias[idx] -= lr * b_grad[idx]
    return model


def test_accuracy(data: pd.DataFrame, model: lib.Model):
    to_predict = data.iloc[:, 0]
    houses = to_predict.unique()
    features = data.iloc[:, 1:]
    right = wrong = 0
    for student in data.index:
        scores = features.loc[student].values
        pred = lib.predict(model, scores)
        proba = lib.softmax(pred)
        result = lib.argmax(proba)
        expected = list(houses).index(to_predict[student])
        if result == expected:
            right += 1
        else:
            wrong += 1

    acc = right / (right + wrong)
    print(f"-- Result --\n{right} correct\n{wrong} incorrect\n {acc} accuracy")


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

    # On peut changer les features a garder ici tant que la premiere colonne
    # est la valeur a prédire ca fonctionne mais ca ameliore pas les resultats
    fields_to_keep = [
        "Hogwarts House",
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Ancient Runes",
        # "Charms",
        # "Flying",
        # "Transfiguration",
        # "History of Magic",
        # "Muggle Studies",
        # "Divination",
        # "Arithmancy",
        # "Potions",
        # "Care of Magical Creatures",
    ]
    data = df.filter(items=fields_to_keep)

    lib.standardisation(data)
    # robust_scaling(data)
    print(data)

    epochs = 35
    lr = 0.015
    print(f"running training for {epochs} epochs with {lr} learning rate")
    model = train(data, lr=lr, epochs=epochs)

    test_accuracy(data, model)
    with open("model.json", "w") as file:
        json.dump(model.__dict__, file)


if __name__ == "__main__":
    main()
