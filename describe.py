import sys
import pandas as pd
import lib


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

    describe(data)


def describe(data: pd.DataFrame):
    numeric = data.select_dtypes(include="number")
    if numeric.empty:
        print("No numeric columns found.")
        sys.exit(1)

    describe = {}
    for col in numeric.columns:
        stats = {}
        val = numeric[col].dropna().tolist()
        quartiles = lib.quartile(val)
        stats["Count"] = len(val)
        stats["Mean"] = lib.mean(val)
        stats["Std"] = lib.std(val)
        stats["Min"] = lib.min(val)
        stats["25%"] = quartiles[0] if quartiles else None
        stats["50%"] = quartiles[1] if quartiles else None
        stats["75%"] = quartiles[2] if quartiles else None
        stats["Max"] = lib.max(val)
        stats["Median"] = lib.median(val)
        stats["Variance"] = lib.var(val)
        stats["Range"] = (
            stats["Max"] - stats["Min"]
            if stats["Max"] is not None and stats["Min"] is not None
            else None
        )
        stats["IQR"] = lib.interquartil_range(val)
        stats["Skewness"] = lib.skewness(val)
        describe[col] = stats
    pd.options.display.float_format = "{:.6f}".format
    print(pd.DataFrame(describe))


if __name__ == "__main__":
    main()
