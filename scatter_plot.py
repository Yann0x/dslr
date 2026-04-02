import sys

import matplotlib.pyplot as plt
import pandas as pd

import describe as yan


def plot_scatter(data: pd.DataFrame, field1, field2, ax):
    ax.clear()

    house_colors = {
        "Ravenclaw": "royalblue",
        "Slytherin": "seagreen",
        "Gryffindor": "firebrick",
        "Hufflepuff": "gold",
    }
    for house, color in house_colors.items():
        try:
            house_data = data[data["Hogwarts House"] == house][
                [field1, field2]
            ].dropna()
        except KeyError as e:
            sys.exit(f"Error: no key {e} ")
        ax.scatter(
            house_data[field1],
            house_data[field2],
            s=10,
            alpha=0.45,
            color=color,
            label=house,
        )
    ax.legend()
    ax.set_xlabel(field1)
    ax.set_ylabel(field2)
    ax.set_title(f"{field1} vs {field2}")


# def normalize_field(col):
#     field = col.to_list()
#     min = yan.my_min(field)
#     max = yan.my_max(field)
#     values = [(x - min) / (max - min) for x in field]
#     print(values)
#     return values


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
    idx1 = 1
    idx2 = 3

    def update(field1, field2):
        plot_scatter(data, field1, field2, ax)
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal idx1, idx2
        if event.key == "right":
            idx2 += 1
            if idx2 >= len(fields):
                idx2 = 0
                idx1 = (idx1 + 1) % len(fields)
            if idx1 == idx2:
                idx2 += 1
        elif event.key == "left":
            idx2 -= 1
            if idx2 < 0:
                idx2 = len(fields) - 1
                idx1 = (idx1 - 1) % len(fields)
            if idx1 == idx2:
                idx2 -= 1
        update(fields[idx1], fields[idx2])

    fig.canvas.mpl_connect("key_press_event", on_key)
    update(fields[idx1], fields[idx2])
    print("Use arrow keys to navigate through features")
    plt.show()
    print(
        "The two features that are the most similar are Atronomy and Defense against the dark Arts"
    )


if __name__ == "__main__":
    main()
