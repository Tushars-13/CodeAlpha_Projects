# Personal Notes - Iris Flower Classification

## Project Goal

Is project ka objective Iris flower ki species predict karna hai using Machine Learning.

Possible species:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

Input Features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

Output:

* Flower Species

Ye ek **Classification Problem** hai kyunki hume categories predict karni hain.

---

# Dataset Loading

```python
import pandas as pd

df = pd.read_csv("Iris.csv")
```

### pd kya hai?

`pd` = pandas ka short name.

Hum likhte hain:

```python
import pandas as pd
```

taaki baar-baar `pandas` na likhna pade.

Example:

```python
pd.read_csv()
```

instead of

```python
pandas.read_csv()
```

---

# DataFrame kya hota hai?

DataFrame ek table jaisa structure hota hai.

Example:

| Name   | Age |
| ------ | --- |
| Tushar | 22  |
| Rahul  | 21  |

Pandas CSV file ko DataFrame me load karta hai.

---

# Important Pandas Commands

## 1. df.head()

```python
df.head()
```

Purpose:

Dataset ki first 5 rows dikhata hai.

Use:

* Data load hua ya nahi
* Columns ka naam check karna

---

## 2. df.shape

```python
df.shape
```

Purpose:

Rows aur columns batata hai.

Example:

```python
(150, 6)
```

Meaning:

* 150 rows
* 6 columns

### Important

Ye function nahi hai.

Correct:

```python
df.shape
```

Wrong:

```python
df.shape()
```

Error:

```python
TypeError: 'tuple' object is not callable
```

Reason:

`shape` ek attribute hai, function nahi.

---

## 3. df.columns

```python
df.columns
```

Purpose:

Saare column names dikhata hai.

Example:

```python
Index([
'Id',
'SepalLengthCm',
'SepalWidthCm',
'PetalLengthCm',
'PetalWidthCm',
'Species'
])
```

---

## 4. df.info()

```python
df.info()
```

Purpose:

Dataset ki summary.

Batata hai:

* Total rows
* Total columns
* Data types
* Missing values

---

## 5. df.describe()

```python
df.describe()
```

Purpose:

Numerical columns ka statistical summary.

Important terms:

### Mean

Average value.

### Min

Smallest value.

### Max

Largest value.

### Std

Standard Deviation.

Data kitna spread hai.

---

## 6. Missing Values Check

```python
df.isnull().sum()
```

Purpose:

Check karta hai ki kisi column me missing values hain ya nahi.

Ideal Output:

```python
0
0
0
0
0
0
```

Meaning:

Dataset clean hai.

---

# Difference Between Attribute and Function

## Attribute

Information return karta hai.

No brackets.

Examples:

```python
df.shape
df.columns
df.dtypes
```

## Function

Kuch operation perform karta hai.

Brackets lagte hain.

Examples:

```python
df.head()
df.info()
df.describe()
```

---

# Machine Learning Workflow

Step 1 → Load Dataset

Step 2 → Understand Dataset

Step 3 → Clean Dataset

Step 4 → Data Visualization

Step 5 → Train Model

Step 6 → Evaluate Model

Step 7 → Make Predictions

Step 8 → Conclusion

---
# Dataset Understanding (EDA - Part 1)
## EDA (Exploratory Data Analysis)

EDA is the process of exploring and understanding a dataset before applying machine learning algorithms.

Objectives of EDA:

- Understand dataset structure
- Check missing values
- Identify data types
- Analyze distributions
- Find relationships between features
- Detect outliers
- Prepare data for modeling

Common EDA Commands:

df.head()
df.shape
df.columns
df.info()
df.describe()
df.isnull().sum()

## Dataset Information

Command:

```python
df.info()
```

Output Summary:

* Total Rows: 150
* Total Columns: 6

Columns:

1. Id
2. SepalLengthCm
3. SepalWidthCm
4. PetalLengthCm
5. PetalWidthCm
6. Species

---

## Data Types

```text
Id                -> int64
SepalLengthCm     -> float64
SepalWidthCm      -> float64
PetalLengthCm     -> float64
PetalWidthCm      -> float64
Species           -> object
```

### Meaning

### int64

Integer values.

Example:

```python
1
2
3
```

Used in:

```python
Id
```

---

### float64

Decimal values.

Example:

```python
5.1
3.5
1.4
```

Used in:

```python
SepalLengthCm
SepalWidthCm
PetalLengthCm
PetalWidthCm
```

---

### object

Text/String data.

Example:

```python
"Iris-setosa"
"Iris-versicolor"
"Iris-virginica"
```

Used in:

```python
Species
```

---

## Non-Null Count

Output:

```text
150 non-null
```

Meaning:

Har row me value present hai.

Koi data missing nahi hai.

---

## Missing Values

Dataset me missing values nahi hain.

Reason:

Har column ka Non-Null Count = 150

Total rows bhi = 150

Therefore:

```text
Missing Values = 0
```

---

# Why Remove Id Column?

Column:

```python
Id
```

Example:

```text
1
2
3
4
5
```

Ye sirf record number hai.

Flower ki species predict karne me iska koi contribution nahi.

Machine Learning model ko is column ki zaroorat nahi hoti.

Therefore:

```python
df = df.drop("Id", axis=1)
```

---

# Understanding axis Parameter

### axis = 0

Rows par operation.

### axis = 1

Columns par operation.

Example:

```python
df.drop("Id", axis=1)
```

Meaning:

"Id" column remove karo.

---

# Feature vs Target

Machine Learning me:

### Features (Input)

Ye values model ko di jati hain.

```python
SepalLengthCm
SepalWidthCm
PetalLengthCm
PetalWidthCm
```

---

### Target (Output)

Jo predict karna hai.

```python
Species
```

---
## Species Distribution Analysis

Command:

df["Species"].value_counts()

Output:

* Iris-setosa = 50
* Iris-versicolor = 50
* Iris-virginica = 50

### Observation

Dataset is perfectly balanced.

Each flower species contains exactly 50 records.

### Importance

Balanced datasets help machine learning models learn all classes fairly and reduce prediction bias.

### Visualization

Code Used:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="Species", data=df)
plt.title("Distribution of Iris Species")
plt.show()
```

Explanation:

* `import seaborn as sns` → Seaborn library ko import karta hai, jo statistical visualizations banane ke liye use hoti hai.
* `import matplotlib.pyplot as plt` → Matplotlib ka plotting module import karta hai.
* `sns.countplot(x="Species", data=df)` → Species column ki har category (flower type) ke records count karke bar chart banata hai.
* `plt.title("Distribution of Iris Species")` → Graph ko title deta hai.
* `plt.show()` → Graph display karta hai.

Result:

All three species appeared with equal frequency (50 samples each).
---
## Pairplot Analysis

Code Used:

```python
sns.pairplot(df, hue="Species")
plt.show()
```

### Purpose

Pairplot is used to visualize relationships between all numerical features and observe class separation.

### Observations

1. Iris-setosa forms a clearly separate cluster from the other two species.

2. Iris-versicolor and Iris-virginica show some overlap, especially in sepal measurements.

3. PetalLengthCm and PetalWidthCm provide the clearest separation among species.

### Conclusion

Petal-related features appear to be the most important predictors for classifying Iris flower species.

The dataset shows good class separability, suggesting that machine learning models should achieve high accuracy.
---
## Correlation Analysis

Command Used:

```python
df.drop("Species", axis=1).corr()
```

### Purpose

Correlation analysis helps identify relationships between numerical features.

Correlation values range from:

* +1 → Strong positive correlation
* 0 → No correlation
* -1 → Strong negative correlation

### Key Observations

1. PetalLengthCm and PetalWidthCm show a very strong positive correlation (0.96).

2. SepalLengthCm and PetalLengthCm also show a strong positive correlation (0.87).

3. SepalWidthCm has moderate negative correlation with PetalLengthCm (-0.42).

4. SepalWidthCm has moderate negative correlation with PetalWidthCm (-0.36).

### Conclusion

Petal-related features are highly correlated and appear to be the most informative features for distinguishing Iris flower species.
---
## Train-Test Split

### Why Train-Test Split?

Machine Learning model ko evaluate karne ke liye dataset ko do parts me divide kiya jata hai:

1. Training Data
2. Testing Data

### Analogy

Training Data = Study Material

Testing Data = Exam

Agar model ko poora dataset dekar usi par test kiya jaye, to evaluation fair nahi hota.

Therefore, some data is kept unseen for testing.

---

## Features and Target

### Features (X)

Input variables used for prediction.

In Iris Dataset:

```python
X = df.drop("Species", axis=1)
```

Features:

* SepalLengthCm
* SepalWidthCm
* PetalLengthCm
* PetalWidthCm

---

### Target (y)

Output variable to be predicted.

```python
y = df["Species"]
```

Target:

* Iris-setosa
* Iris-versicolor
* Iris-virginica

---

## Train-Test Split Code

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

## Understanding Parameters

### test_size = 0.2

20% of data will be used for testing.

Dataset Size:

```text
Total Rows = 150
```

Calculation:

```text
20% of 150 = 30 rows
```

Therefore:

```text
Training Rows = 120
Testing Rows = 30
```

---

### random_state = 42

Used to make train-test splitting reproducible.

Without random_state:

* Different split every run

With random_state:

* Same split every run

Benefits:

* Consistent results
* Easy debugging
* Reproducible experiments

---

## Output Variables

### X_train

Training feature data.

Used to teach the model.

Expected Shape:

```text
(120, 4)
```

---

### X_test

Testing feature data.

Used to evaluate the model.

Expected Shape:

```text
(30, 4)
```

---

### y_train

Correct species labels for training data.

---

### y_test

Correct species labels for testing data.

---
## KNN Model Training and Evaluation

### Algorithm Used

K-Nearest Neighbors (KNN)

### Model Creation

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
```

### Model Training

```python
model.fit(X_train, y_train)
```

Purpose:

The model learns patterns from the training dataset.

### Prediction

```python
y_pred = model.predict(X_test)
```

Purpose:

Predict species labels for unseen test data.

### Accuracy Calculation

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
```

### Result

Accuracy = 1.0

Accuracy Percentage = 100%

### Interpretation

The model correctly classified all test samples.

### Possible Reasons for High Accuracy

* Dataset contains no missing values.
* Dataset is balanced across all species.
* Iris-setosa is clearly separable from other species.
* Petal measurements provide strong discriminatory power.

### Conclusion

The KNN model performed exceptionally well on the Iris dataset and achieved perfect classification accuracy on the test set.
---
## Classification Report Analysis

Code Used:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

### Important Metrics

#### Precision

Measures how many predicted samples of a class were actually correct.

Formula:

Precision = Correct Predictions / Total Predicted Samples

Result:

1.00 for all species.

---

#### Recall

Measures how many actual samples of a class were correctly identified.

Formula:

Recall = Correct Predictions / Actual Samples

Result:

1.00 for all species.

---

#### F1-Score

Harmonic mean of Precision and Recall.

Used when both metrics are important.

Result:

1.00 for all species.

---

#### Support

Number of actual test samples belonging to each class.

Results:

* Iris-setosa = 10
* Iris-versicolor = 9
* Iris-virginica = 11

Total Test Samples = 30

---

### Overall Result

Accuracy = 100%

The KNN classifier correctly classified all test samples.

No misclassification was observed in the testing dataset.

### Conclusion

The Iris dataset is highly separable, especially using petal measurements. This allowed the KNN classifier to achieve perfect performance on the test data.
---
