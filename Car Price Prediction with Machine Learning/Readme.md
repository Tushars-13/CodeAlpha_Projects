# 🚗 Car Price Prediction using Machine Learning

## 📌 Project Overview

This project aims to predict the selling price of used cars using Machine Learning techniques. The model analyzes various car-related features such as manufacturing year, current market price, kilometers driven, fuel type, transmission type, and ownership history to estimate the car's selling price.

The project demonstrates a complete Machine Learning workflow including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and visualization.

---

## 🎯 Objective

The main objectives of this project are:

* Understand and analyze a real-world car dataset.
* Perform Exploratory Data Analysis (EDA).
* Identify important factors affecting car prices.
* Build and compare multiple regression models.
* Predict the selling price of used cars.
* Evaluate model performance using appropriate metrics.

---

## 📂 Dataset Information

### Features

* **Car_Name** *(removed during preprocessing)*
* **Year** – Manufacturing year of the car
* **Present_Price** – Current showroom price (in lakhs)
* **Driven_kms** – Total kilometers driven
* **Fuel_Type** – Petrol, Diesel, or CNG
* **Selling_type** – Dealer or Individual
* **Transmission** – Manual or Automatic
* **Owner** – Number of previous owners

### Target Variable

* **Selling_Price** – Selling price of the used car (in lakhs)

### Dataset Summary

* Total Records: **301**
* Multiple car brands and models
* Mixed numerical and categorical features

---

## 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

* Dataset inspection
* Missing value analysis
* Correlation analysis
* Heatmap visualization
* Fuel Type vs Selling Price analysis
* Transmission vs Selling Price analysis
* Feature importance analysis

### Key Findings

* No missing values were present in the dataset.
* Present_Price showed the strongest correlation with Selling_Price.
* Newer cars generally had higher selling prices.
* Cars with lower mileage tended to have better resale value.
* Fuel Type and Transmission had comparatively smaller influence on predictions.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were applied:

### Feature Removal

* Removed **Car_Name** column as it contained many unique values and contributed little to prediction performance.

### Encoding

* **Transmission**

  * Manual → 0
  * Automatic → 1

* **Selling_type**

  * Dealer → 0
  * Individual → 1

* **Fuel_Type**

  * One-Hot Encoding applied using `pd.get_dummies()`

### Train-Test Split

* Training Data: **80%**
* Testing Data: **20%**

---

## 🤖 Machine Learning Models Used

### 1. Linear Regression

A baseline regression model used to establish initial performance.

**R² Score:** 0.8489

---

### 2. Random Forest Regressor

An ensemble learning algorithm that combines multiple decision trees for improved prediction performance.

**R² Score:** 0.9646

---

## 📊 Model Performance

| Model                   | R² Score |
| ----------------------- | -------- |
| Linear Regression       | 0.8489   |
| Random Forest Regressor | 0.9646   |

### Best Model

🏆 **Random Forest Regressor**

The Random Forest model significantly outperformed Linear Regression and achieved excellent predictive performance.

---

## 📈 Visualizations

The following visualizations were created:

* Correlation Heatmap
* Fuel Type vs Selling Price Boxplot
* Transmission vs Selling Price Boxplot
* Feature Importance Graph
* Actual vs Predicted Price Scatter Plot

### Feature Importance

The most influential features were:

1. Present_Price
2. Year
3. Driven_kms

These features had the greatest impact on car price prediction.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook

---

## 📁 Project Structure

```text
car-price-prediction/
│
├── car_price_prediction.ipynb
├── car data.csv
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <repository-link>
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 3. Open Jupyter Notebook

```bash
jupyter notebook
```

### 4. Run the Notebook

Open:

```text
car_price_prediction.ipynb
```

and run all cells.

---

## 📚 Learning Outcomes

Through this project, I learned:

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Correlation Analysis
* Feature Engineering
* Categorical Data Encoding
* Train-Test Split
* Linear Regression
* Random Forest Regression
* Model Evaluation using R² Score
* Feature Importance Analysis
* Data Visualization

---

## 👨‍💻 Author

**Tushar Shukla**



