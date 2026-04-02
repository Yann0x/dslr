import sys

import pandas as pd

import describe as yann


def interquartil_range(values: list):
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
                # data.loc[i, field] = mean
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

    yann.describe(data)
    robust_scaling(data)
    yann.describe(data)


if __name__ == "__main__":
    main()
