# This model:
1. Uses the dataset disease.csv, which has 2 classes (cholesterol_level, has_heart_disease)
2. Is trained by a Logistic Regression model to predict if a patient has heart disease based on his/her cholesterol levels
3. Has evaluation metrics for the model

**Note:**
1. Probability represented as 1 (patient very likely to have heart disease) and 0 (patient very unlikely to have heart disease)
2. Predicted probabilities are plotted for both training and test sets to visualize how well the logistic regression model is performing

## Files
1. "disease.csv" is the dataset, which will give you an accuracy of only 0.5333333333333333.  That's because we're training the model on only one (1) feature, a.k.a cholesterol_level. #noobie :)
2. "disease.py" is the model
3. "Assets" is the folder containing screenshots of the bs predicted model
