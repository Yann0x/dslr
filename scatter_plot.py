import sys

import matplotlib.pyplot as plt
import pandas as pd

import describe as yan


def plot_scatter(values):

    plt.scatter(range(len(values)), values, s=5, alpha=0.5)


def normalize_field(col):
    field = col.to_list()
    min = yan.my_min(field)
    max = yan.my_max(field)
    values = [(x - min) / (max - min) for x in field]
    print(values)
    return values


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <dataset.csv>")
        sys.exit(1)

    try:
        data = pd.read_csv(sys.argv[1])
    except FileNotFoundError:
        print(f"Error: file '{sys.argv[1]}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    means = []
    for field in data.columns:
        if (
            not pd.api.types.is_numeric_dtype(data[field])
            or data[field].empty
            or field == "Index"
        ):
            continue
        values = normalize_field(data[field])
        means.append(yan.median(values))
    plot_scatter(means)

    plt.show()


if __name__ == "__main__":
    main()
