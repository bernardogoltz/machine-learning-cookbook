# Classification
- ```sklearn 1.3.0```
- ```Python 3.11.5```

## 1. Binary Classifier
- Categorize into one of two categories

Label data and Target -> Train/Test split  

### Evaluating model's performance 

```python
from sklearn.metrics import precision_score , recall_score 
precision_score(Y_train_n , y_train_pred)
recall_score(Y_train_n , y_train_pred)
```

```python
from sklearn.metrics import classification_report
classification_report(Y_train_n , y_train_pred)

"""
------------------------------------------------------
              precision    recall  f1-score   support

       False       0.99      1.00      1.00     43501
        True       0.99      0.94      0.96      5499

    accuracy                           0.99     49000
   macro avg       0.99      0.97      0.98     49000
weighted avg       0.99      0.99      0.99     49000
-------------------------------------------------------
"""
```
