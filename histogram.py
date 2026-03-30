import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_histogram(data: pd.DataFrame, column: str):

    field = data[column]
    field.dropna()
    if not pd.api.types.is_numeric_dtype(field) or field.empty:
        print(f"Column {column} is non numeric, skipping it...")
        return

    plt.hist(field)
    plt.show()


def main():
    data = pd.read_csv("./datasets/dataset_train.csv")
    # Source - https://stackoverflow.com/a/79037727
    # Posted by WhaSukGO
    # Retrieved 2026-03-30, License - CC BY-SA 4.0

    for column in data.columns:
        plot_histogram(data, column)


if __name__ == "__main__":
    main()
