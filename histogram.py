import matplotlib.pyplot as plt
import pandas as pd


def plot_histogram(data: pd.DataFrame, field: str):
    column = data[field]
    column.dropna()
    if not pd.api.types.is_numeric_dtype(column) or column.empty or field == "Index":
        print(f"Column {field} is non relevant, skipping it...")
        return
    house_colors = {
        "Ravenclaw": "royalblue",
        "Slytherin": "seagreen",
        "Gryffindor": "firebrick",
        "Hufflepuff": "gold",
    }
    for house, color in house_colors.items():
        house_values = data.loc[data["Hogwarts House"] == house, field].dropna()
        if house_values.empty:
            continue
        plt.hist(house_values, bins=30, alpha=0.45, color=color, label=house)

    plt.title(field)
    plt.legend()
    plt.show()


# def plot_histogram(scores: dict):

#     min = max = scores[next(iter(scores))]
#     for value in scores.values():
#         if value > max:
#             max = value
#         if value < min:
#             min = value

#     plt.bar(*zip(*scores.items()))
#     # plt.ylim(min, max)
#     # plt.tight_layout()
#     plt.show()


def get_scores(data: pd.DataFrame, field: str, house: str):
    values = data.loc[data["Hogwarts House"] == house, field].dropna()
    res = values.sum()
    print(res)
    return res


def main():
    data = pd.read_csv("./datasets/dataset_train.csv")

    for field in data.columns:
        plot_histogram(
            data,
            field,
        )


if __name__ == "__main__":
    main()
