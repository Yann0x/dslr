from matplotlib.axes import Axes
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import sys


def add_histograms(data: pd.DataFrame, axes):
    house_colors = {
        "Ravenclaw": "royalblue",
        "Slytherin": "seagreen",
        "Gryffindor": "firebrick",
        "Hufflepuff": "gold",
    }
    i: int = 0
    bins = 30
    numeric_raw = data.select_dtypes(include="number")
    if "Index" in numeric_raw.columns:
        numeric_raw = numeric_raw.drop("Index", axis=1)
    numeric = numeric_raw.dropna()
    for element in numeric:
        ax: Axes = axes[i][i]
        for house, color in house_colors.items():
            try:
                house_values = data.loc[data["Hogwarts House"] == house, element].dropna()
            except KeyError as e:
                sys.exit(f"Error: no key {e} ")
            if house_values.empty:
                continue
            ax.hist(house_values, alpha=0.45, bins=bins, color=color, label=house)
        ax.set_title(str(element))
        ax.legend()
        i += 1


def add_versus(data: pd.DataFrame, axes):
    house_colors = {
        "Ravenclaw": "royalblue",
        "Slytherin": "seagreen",
        "Gryffindor": "firebrick",
        "Hufflepuff": "gold",
    }
    numeric_raw = data.select_dtypes(include="number")
    if "Index" in numeric_raw.columns:
        numeric_raw = numeric_raw.drop("Index", axis=1)
    numeric = numeric_raw.dropna()
    i: int = 0
    for element in numeric:
        for j in range(len(numeric.columns)):
            if i != j:
                ax: Axes = axes[i][j]
                for house, colors in house_colors.items():
                    house_values = data.loc[data["Hogwarts House"] == house, [str(element), numeric.columns[j]]].dropna()
                    ax.scatter(house_values[element].to_list(), house_values[numeric.columns[j]].to_list(), color=colors, alpha=0.45, s=9)
                ax.set_title(f"{str(element).split()[0]} vs {numeric.columns[j].split()[0]}")
        i+= 1


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

    if "Hogwarts House" not in data.columns:
        print("Error: 'Hogwarts House' column not found.")
        sys.exit(1)
    numeric_raw = data.select_dtypes(include="number")
    if "Index" in numeric_raw.columns:
        numeric_raw = numeric_raw.drop("Index", axis=1)
    numeric = numeric_raw.dropna()
    if numeric.empty:
        print("No numeric columns found.")
        sys.exit(1)
    fig, axes = plt.subplots(len(numeric.dtypes), len(numeric.dtypes), figsize = (50,50))
    add_histograms(data, axes)
    add_versus(data, axes)
    plt.savefig("pair_plot.png", dpi=150)
    return


if __name__ == "__main__":
    main()