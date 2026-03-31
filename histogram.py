import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import RadioButtons


def plot_histogram(data: pd.DataFrame, field: str, ax):
    ax.clear()
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
        ax.hist(house_values, bins=bins, alpha=0.45, color=color, label=house)
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

    ax.set_title(f"{field} ({variabilite_indice:.2f})")
    ax.legend()
    return plot_histogram.static_var


def get_scores(data: pd.DataFrame, field: str, house: str):
    values = data.loc[data["Hogwarts House"] == house, field].dropna()
    res = values.sum()
    return res


def main():
    data = pd.read_csv("./datasets/dataset_train.csv")

    # build the fields list:
    fields = []
    for field in data.columns:
        if (
            not pd.api.types.is_numeric_dtype(data[field])
            or data[field].empty
            or field == "Index"
        ):
            continue
        fields.append(field)

    fig, ax = plt.subplots()

    best = None

    def update(field):
        nonlocal best
        best = plot_histogram(data, field, ax)
        fig.canvas.draw_idle()

    current_idx = current_idx = 10

    def on_key(event):
        nonlocal current_idx
        if event.key == "right":
            current_idx = (current_idx + 1) % len(fields)
        elif event.key == "left":
            current_idx = (current_idx - 1) % len(fields)
        update(fields[current_idx])

    fig.canvas.mpl_connect("key_press_event", on_key)

    for field in fields:
        update(field)
    update(fields[10])

    print("Use arrow keys to navigate courses")
    plt.show()

    print(
        f"The course that has an homogoneous distribution across houses is {best[1] if best else 'unknown'}"
    )


if __name__ == "__main__":
    main()
