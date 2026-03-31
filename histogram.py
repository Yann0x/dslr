import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_histogram(data: pd.DataFrame, field: str):
    column = data[field]
    column.dropna()
    if column.empty:
        return
    house_colors = {
        "Ravenclaw": "royalblue",
        "Slytherin": "seagreen",
        "Gryffindor": "firebrick",
        "Hufflepuff": "gold",
    }
    bins = 30
    hist = []
    for house, color in house_colors.items():
        house_values = data.loc[data["Hogwarts House"] == house, field].dropna()
        if house_values.empty:
            continue
        plt.hist(house_values, bins=bins, alpha=0.45, color=color, label=house)
        hist.append(np.histogram(house_values, bins=bins)[0])

    trsp_hist = list(zip(*hist))
    variabilite_indices = []

    for bin in range(len(trsp_hist)):
        bin_values = [row for row in trsp_hist[bin]]
        variabilite_indices.append(np.var(bin_values))

    variabilite_indice = sum(variabilite_indices) / bins
    if not hasattr(plot_histogram, "static_var"):
        plot_histogram.static_var = (variabilite_indice, field)
    if variabilite_indice < plot_histogram.static_var[0]:
        plot_histogram.static_var = (variabilite_indice, field)

    plt.title(f"{field} ({variabilite_indice:.2f})")
    plt.legend()
    plt.show()
    return plot_histogram.static_var


def get_scores(data: pd.DataFrame, field: str, house: str):
    values = data.loc[data["Hogwarts House"] == house, field].dropna()
    res = values.sum()
    return res


def main():
    data = pd.read_csv("./datasets/dataset_train.csv")

    best = None
    for field in data.columns:
        if (
            not pd.api.types.is_numeric_dtype(data[field])
            or data[field].empty
            or field == "Index"
        ):
            continue
        best = plot_histogram(
            data,
            field,
        )
        if best is not None:
            print(f"Current best : {best[1]} with score {best[0]:.2f}")
    print(
        f"The course that has an homogoneous distribution across houses is {best[1] if best else 'unknown'}"
    )


if __name__ == "__main__":
    main()
