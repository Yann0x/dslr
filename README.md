# DSLR - Data Science x Logistic Regression

> **42 School Project** - Implement a logistic regression classifier from scratch to predict which Hogwarts house a student belongs to, based on their course scores. No sklearn.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Step 1 - Data Description (`describe.py`)](#3-step-1---data-description-describepy)
4. [Step 2 - Histogram (`histogram.py`)](#4-step-2---histogram-histogrampy)
5. [Step 3 - Scatter Plot (`scatter_plot.py`)](#5-step-3---scatter-plot-scatter_plotpy)
6. [Step 4 - Pair Plot (`pair_plot.py`)](#6-step-4---pair-plot-pair_plotpy)
7. [Step 5 - Training (`logreg_train.py`)](#7-step-5---training-logreg_trainpy)
8. [Step 6 - Prediction (`logreg_predict.py`)](#8-step-6---prediction-logreg_predictpy)
9. [Math Behind It](#9-math-behind-it)
10. [Installation & Usage](#10-installation--usage)

---

## 1. Project Overview

The goal is to replicate what `pandas.describe()` and `sklearn` do, without using them for the heavy lifting. Everything is computed from scratch:

- Statistical descriptors (mean, std, quartiles, skewness...)
- Data visualizations (histograms, scatter plots, pair plots)
- Logistic regression with gradient descent (Batch, Mini-Batch, SGD)
- Multi-class classification using **Softmax**

The classifier assigns each student to one of the four Hogwarts houses:

| House | Color |
|-------|-------|
| Gryffindor | Red |
| Slytherin | Green |
| Ravenclaw | Blue |
| Hufflepuff | Gold |

---

## 2. Dataset

Two CSV files are provided under `datasets/`:

| File | Rows | Purpose |
|------|------|---------|
| `dataset_train.csv` | 1600 | Training - contains the `Hogwarts House` label |
| `dataset_test.csv` | ~400 | Prediction - no label, the model must assign one |

Each row is a student with 13 numerical course scores:

```
Arithmancy, Astronomy, Herbology, Defense Against the Dark Arts,
Divination, Muggle Studies, Ancient Runes, History of Magic,
Transfiguration, Potions, Care of Magical Creatures, Charms, Flying
```

Some values are `NaN` (missing). The model handles them during normalization.

---

## 3. Step 1 - Data Description (`describe.py`)

### Goal

Reproduce `pandas.describe()` using only custom math functions from `lib.py`. No numpy, no scipy for computations.

### What it computes

| Stat | Description |
|------|-------------|
| Count | Number of non-null values |
| Mean | Arithmetic average |
| Std | Standard deviation (sample, ddof=1) |
| Min / Max | Extremes |
| 25% / 50% / 75% | Quartiles via linear interpolation |
| Median | Middle value |
| Variance | Sample variance |
| Range | Max - Min |
| IQR | Q3 - Q1 |
| Skewness | Third standardized moment |

### Run

```bash
python describe.py datasets/dataset_train.csv
```

### Output (truncated)

```
                 Index       Arithmancy  ...      Charms      Flying
Count      1600.000000      1566.000000  ... 1600.000000 1600.000000
Mean        799.500625     49634.570243  ... -243.374409   21.958012
Std         462.023458     16679.806036  ...    8.783640   97.631602
Min           0.000000    -24370.000000  ... -261.048920 -181.470000
25%         399.750000     38511.500000  ... -250.652600  -41.870000
50%         799.500000     49013.500000  ... -244.867765   -2.515000
75%        1199.250000     60811.250000  ... -232.552305   50.560000
Max        1599.000000    104956.000000  ... -225.428140  279.070000
Median      799.500000     49013.500000  ... -244.867765   -2.515000
Variance 213465.676047 278215929.383881  ...   77.152329 9531.929722
Range      1599.000000    129326.000000  ...   35.620780  460.540000
IQR         799.500000     22299.750000  ...   18.100295   92.430000
Skewness      0.000008        -0.041879  ...    0.390012    0.882497
```

---

## 4. Step 2 - Histogram (`histogram.py`)

### Goal

Visualize the score distribution of each course split by house, to find which course has the most similar distribution across all houses (least useful for classification).

For each course, 4 overlapping semi-transparent histograms are drawn. A variance score across bins is computed: the lower it is, the more homogeneous the distribution. Use arrow keys (`<- ->`) to navigate courses interactively.

### Run

```bash
python histogram.py datasets/dataset_train.csv
```

### Result

The course with the most homogeneous distribution is **Care of Magical Creatures**.

![Histogram - Care of Magical Creatures](img/img_histogram.png)

![Histogram grid - all courses](img/img_histogram_all.png)

Courses like **Astronomy**, **Herbology**, and **Defense Against the Dark Arts** show clearly separated distributions between houses and are the most useful for classification.

---

## 5. Step 3 - Scatter Plot (`scatter_plot.py`)

### Goal

Find the two features that are most correlated by plotting each pair as a 2D scatter plot. Each point is a student, colored by house. Navigate all pairs interactively with arrow keys.

### Run

```bash
python scatter_plot.py datasets/dataset_train.csv
```

### Result

**Astronomy** and **Defense Against the Dark Arts** are near-perfectly correlated - their scatter plot forms a single straight line, meaning they carry nearly identical information.

![Scatter Plot - Astronomy vs Defense Against the Dark Arts](img/img_scatter.png)

---

## 6. Step 4 - Pair Plot (`pair_plot.py`)

### Goal

Display all pairwise feature combinations at once. Diagonal cells show histograms per house, off-diagonal cells show scatter plots. This is what drives the feature selection for training.

### Run

```bash
python pair_plot.py datasets/dataset_train.csv
```

This saves a `pair_plot.png` file (50x50 inches at 150 DPI).

### Result

![Pair Plot](img/img_pair_plot.png)

Each cell at `(row i, col j)` plots `feature_i` vs `feature_j` colored by house. A useful pair shows 4 distinct color clusters. A useless one is a single mixed blob.

#### Features to keep

The following pairs all show clearly separated clusters per house:

- Astronomy x Herbology
- Astronomy x Ancient Runes
- Herbology x Defense Against the Dark Arts
- Herbology x Ancient Runes
- Defense Against the Dark Arts x Ancient Runes

These 4 features - **Astronomy, Herbology, Defense Against the Dark Arts, Ancient Runes** - are selected for training.

#### Redundant pair

**Astronomy x Defense Against the Dark Arts** show a perfect anti-diagonal line, with a near-perfect negative correlation (approx. -1). Dropping either one would not change the model's accuracy.

#### Features to ignore

All remaining features (Arithmancy, Divination, History of Magic, Transfiguration, Muggle Studies, Potions, Flying, Charms, Care of Magical Creatures) show fully mixed house colors in every scatter cell and are dropped.

---

## 7. Step 5 - Training (`logreg_train.py`)

### Goal

Train a multi-class logistic regression model from scratch.

### Features used

```python
fields_to_keep = [
    "Hogwarts House",
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Ancient Runes",
]
```

### Normalization

All features are standardized before training (zero mean, unit variance):

```
x_normalized = (x - mean) / std
```

Missing values (`NaN`) are replaced by `0` (the mean after standardization).

### Model

4 linear classifiers, one per house, each with weights `W` and bias `b`:

```
score_house_k = W_k . x + b_k
```

Softmax converts the 4 raw scores into probabilities:

```
P(house_k | x) = exp(score_k) / sum(exp(score_j))
```

The predicted house is the one with the highest probability.

### Gradient Descent

Three algorithms are available:

| Algorithm | Flag | Description |
|-----------|------|-------------|
| Batch GD | *(default)* | Gradients computed on the full dataset per epoch |
| Mini-Batch GD | `MINI_BATCH` | Batches of 32 students |
| SGD | `SGD` | Update after each individual student |

Cross-entropy gradient for each class `k`:

```
grad_W_k = (P(k|x) - y_k) . x
grad_b_k =  P(k|x) - y_k
```

Where `y_k = 1` if the student belongs to house `k`, else `0`.

### Hyperparameters

```python
epochs = 50
learning_rate = 0.015
```

### Run

```bash
python logreg_train.py datasets/dataset_train.csv
python logreg_train.py datasets/dataset_train.csv MINI_BATCH
python logreg_train.py datasets/dataset_train.csv SGD
```

### Training Accuracy Curve

![Accuracy over epochs](img/img_accuracy.png)

The model reaches **~98.19% accuracy** within the first few epochs.

### Output

The trained model is saved as `model.json`:

```json
{
  "weights": [[...], [...], [...], [...]],
  "bias": [0.0, 0.0, 0.0, 0.0],
  "house_field": "Hogwarts House",
  "houses_names": ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"],
  "features": ["Astronomy", "Herbology", "Defense Against the Dark Arts", "Ancient Runes", "Care of Magical Creatures"],
  "normalisation_method": "standardisation"
}
```

---

## 8. Step 6 - Prediction (`logreg_predict.py`)

Loads `model.json`, applies the same normalization to the test dataset, and writes a house prediction for every student to `houses.csv`.

### Run

```bash
python logreg_predict.py datasets/dataset_test.csv
```

### Output

```
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
3,Slytherin
...
```

---

## 9. Math Behind It

### Softmax

Converts raw scores into a probability distribution over K classes:

$$P(k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

### Cross-Entropy Loss

For one sample:

$$L = -\log P(y \mid x)$$

The gradient of this loss with respect to the weights gives exactly `(P(k|x) - y_k) . x`.

### Standardization

$$x' = \frac{x - \mu}{\sigma}$$

Puts all features on the same scale, which is required for gradient descent to converge properly.

---

## 10. Installation & Usage

```bash
pip install -r requirements.txt
```

```bash
python describe.py datasets/dataset_train.csv
python histogram.py datasets/dataset_train.csv
python scatter_plot.py datasets/dataset_train.csv
python pair_plot.py datasets/dataset_train.csv
python logreg_train.py datasets/dataset_train.csv
python logreg_predict.py datasets/dataset_test.csv
```

### File structure

```
dslr/
├── datasets/
│   ├── dataset_train.csv
│   └── dataset_test.csv
├── lib.py
├── describe.py
├── histogram.py
├── scatter_plot.py
├── pair_plot.py
├── logreg_train.py
├── logreg_predict.py
├── model.json          (generated after training)
└── houses.csv          (generated after prediction)
```

---

*42 School - Data Science x Logistic Regression*
