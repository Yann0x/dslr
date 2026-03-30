import sys
import pandas as pd


def mean(args: list) -> float | None:
    """Compute the arithmetic mean of the given values."""
    if not args:
        return None
    try:
        res = sum(elem for elem in args) / len(args)
    except Exception:
        return None
    return res


def median(args: list) -> float | int | None:
    """Compute the median of the given values."""
    if not args:
        return None
    try:
        args_list = sorted(args)
    except Exception:
        return None
    argssize = len(args_list) // 2
    if len(args_list) % 2 == 0:
        median = (args_list[argssize - 1] + args_list[argssize]) / 2
    else:
        median = args_list[argssize]
    return median


def quartile(args: list) -> list[float] | None:
    """Compute the first , second and third quartiles of the given values."""
    if not args:
        return None
    try:
        args_list = sorted(args)
    except Exception:
        return None
    argssize = len(args_list) - 1
    quarts = [argssize / 4, argssize / 2, argssize * 3 / 4]
    res = []
    for i in range(3):
        if quarts[i] != int(quarts[i]):
            index = int(quarts[i])
            rest = quarts[i] - int(quarts[i])
            res.append(args_list[index] * (1 - rest)
                       + args_list[index + 1] * rest)
        else:
            res.append(args_list[int(quarts[i])])
    return res


def my_var(args: list) -> float | int | None:
    """Compute the variance of the given values."""
    if len(args) < 2:
        return None
    meaning = mean(args)
    if meaning is not None:
        variance = sum((x - meaning) ** 2 for x in args) / (len(args) - 1)
        return variance
    return None


def my_std(args: list) -> float | int | None:
    """Compute the standard deviation of the given values."""
    variance = my_var(args)
    if variance is not None:
        return variance**0.5
    return None


def my_min(args: list) -> float | int | None:
    """Returns the smallest value in the args"""
    if not args:
        return None
    min_val = args[0]
    for elem in args:
        if elem < min_val:
            min_val = elem
    return min_val


def my_max(args: list) -> float | int | None:
    """Returns the largest value in the args"""
    if not args:
        return None
    max_val = args[0]
    for elem in args:
        if elem > max_val:
            max_val = elem
    return max_val


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

    numeric = data.select_dtypes(include="number")
    if numeric.empty:
        print("No numeric columns found.")
        sys.exit(1)

    describe = {}
    for col in numeric.columns:
        stats = {}
        val = numeric[col].dropna().tolist()
        quartiles = quartile(val)
        stats["Count"] = len(val)
        stats["Mean"] = mean(val)
        stats["Std"] = my_std(val)
        stats["Min"] = my_min(val)
        stats["25%"] = quartiles[0] if quartiles else None
        stats["50%"] = quartiles[1] if quartiles else None
        stats["75%"] = quartiles[2] if quartiles else None
        stats["Max"] = my_max(val)
        describe[col] = stats
    print(pd.DataFrame(describe))


if __name__ == "__main__":
    main()
