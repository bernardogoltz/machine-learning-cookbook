# Machine Learning: _sklearn_ general approach

## 1 - Instaling Requirements

```bash
# using pip
$pip install -U scikit-learn
```

```bash
# using conda
$conda create --name sklearn-sandbox
$conda activate sklearn-sandbox
$conda install scikit-learn
```

## 2 - Loading, _feature/target_ and _train/test_ split

### 2.1. Import Libraries and Feature/Target separation
```python
from sklearn.model_selection import train_test_split

# Example Data
from sklearn import datasets
diabetes = datasets.load_diabetes()

# should be a little more complicated than that , but the general idea
# is to separate the features to the target variables

X = diabetes.data
Y = diabetes.target
```

### 2.2. Perform 80/20 data split
```python
X_train , X_test, Y_train, Y_test = train_test_split(X, Y , test_size = 0.2)
```

### 2.3 Train

```python
from sklearn import linear_model
model = linear_model.LinearRegression()

model.fit(X_train , Y_train)
```

### 2.4 Test
Perform a numeric comparison between the ESTIMATED and ACTUAL value of target labels of our data.
```python
from sklearn.metrics import mean_squared_error , r2_score
Y_pred = model.predict(X_test)

MSE = mean_squared_error(Y_test , Y_pred)
R2 = r2_score(Y_test, Y_test)
```
#### Useful Parameters of the model's prediction: 
```bash
model.coef_ # weight values of each feature
model.intercept_ # y intercept
```

## References 
[1] [Scikit-learn](https://scikit-learn.org/stable/install.html): Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

