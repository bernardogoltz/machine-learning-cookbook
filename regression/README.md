# Regression General Approach 
- ```sklearn 1.3.0```
- ```Python 3.11.5```

## 1. Handling Categorical Variables 
### 1.1 Ordinal Encoder 
```python
# A numerical value for each ordinary class in the column
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
encoded_variable = ordinal_encoder.fit_transform(df[["Categorical Variable"]])
```
### 1.2 One Hot Encoder
```python
# With several classes, models may be biased with ordinal values 
# one hot create one column for each class. 

from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()
encoded_matrix = one_hot_encoder.fit_transform(df[["Categorical Variable"]])
```



## 2. Train/Test Split + Linear Regression Fit + Model Evaluating
```python
# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(test_size = 0.2)
```

### 2.2 Linear Regression
```python
# Training the linear regression model
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(X_train, Y_train)

train_predicitons = linear_regression.fit(X_train)
```

### 2.3 Scores
```python
from sklearn.metrics import mean_squared_error 
lin_mse = mean_squared_error(Y_train , predicions)
```

## 3. Cross Validation
Evaluation for novel data.
```python
from sklearn.model_selection import cross_val_score

scores =  cross_val_score(linear_regression, X_train , Y_train , scoring = "neg_mean_squared_error" , cv = 10) 

linear_rmse_scores = np.sqrt(-scores)
```

## 4. GridSearch
For testing several hyperparameters combinations
```python
from sklearn.model_selection import GridSearchCV
# a list containing dictionaries
param_grid = [
    {'parameter_1':[x1 , x2 , x3], 'parameter_2':[w1 , w2 , w3]}, # 9 combinations
    {'parameter_1':[z1 , z2 , z3] , 'parameter_1':[y1 , y2]} # 6 combinations
]

grid_search = GridSearchCV(
        model , 
        param_grid ,
        cv = 5 ,
        scoring = 'neg_mean_squared_error'
        )
```
Now, based on our tests with grid search we can assume a "final model" for our data, called bellow as ```best_model``` 
```IPython
# best params founded in combinations of the grid
grid_search.best_estimator_
grid_search.best_params_ 

best_model = grid_search.best_estimator_

```

## 5. Testing the final model score

```Python
# Final model founded on grid search

best_model_predictions = best_model.predict(X_test)

MSE = mean_squared_error(Y_test , best_model_predictions) 
```