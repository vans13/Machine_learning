from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
  mean_absolute_error,
  mean_squared_error,
  accuracy_score,
  r2_score,
  f1_score,
  confusion_matrix,
  classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Sequence

def function_split(X: pd.DataFrame, y: np.ndarray, rs: int) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
  """
  summary:
    Split features and labels into training and testing sets using an 80/20 ratio.
  args:
    X: Feature matrix (Pandas DataFrame or array-like) to split.
    y: Target vector (array-like) aligned with X.
    rs: Random seed to ensure reproducibility of the split.
  returns:
    X_train: Feature matrix for training.
    X_test: Feature matrix for testing.
    y_train: Target vector for training.
    y_test: Target vector for testing.
  """
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, train_size=0.8, test_size=0.2, random_state=rs, stratify=y
  )
  return X_train, X_test, y_train, y_test

def function_standard_scaler(X_train, X_test):
  """Scale features to zero mean and unit variance using StandardScaler.
    args:
      X_train: Training feature matrix.
      X_test: Testing feature matrix.
    returns:
      X_train_scaled: Standardized training features.
      X_test_scaled: Standardized testing features.
  """

  scaler=StandardScaler()
  X_train_scaled=scaler.fit_transform(X_train)
  X_test_scaled=scaler.transform(X_test)

  return X_train_scaled, X_test_scaled

def function_min_max_scaler(X_train, X_test):
  """Scale features to a fixed range [0, 1] using MinMaxScaler.
    args:
      X_train: Training feature matrix.
      X_test: Testing feature matrix.
    returns:
      X_train_scaled: Min-max normalized training features.
      X_test_scaled: Min-max normalized testing features.
  """
  scaler=MinMaxScaler()
  X_train_scaled=scaler.fit_transform(X_train)
  X_test_scaled=scaler.transform(X_test)

  return X_train_scaled, X_test_scaled

def function_fit (X_train_scaled, y_train):
  """Train a Linear Regression model on scaled features.
    args:
      X_train_scaled: Scaled training features.
      y_train: Training targets (class indices 0, 1, 2 for Iris).
    returns:
      model_iris: Fitted LinearRegression instance.
  """
  model_iris=LinearRegression()
  model_iris.fit(X_train_scaled, y_train)

  return model_iris

def function_predict(model_iris, X_test_scaled):
  """Generate predictions for the test set using a trained model.
    args:
      model_iris: Trained LinearRegression model.
      X_test_scaled: Scaled testing features.
    returns:
      y_pred_iris: Continuous predictions (floats) from the regression model.
  """
  y_pred_iris= model_iris.predict(X_test_scaled)

  return y_pred_iris

def regression_to_class(y_pred_iris):
  """Convert continuous predictions into discrete class labels 0, 1, or 2.
    args:
      y_pred_iris: Continuous predictions (floats) from LinearRegression.
    returns:
      y_pred_iris_class: Integer class labels in the range [0, 2].
  """
  y_pred_iris_class = np.rint(y_pred_iris).astype(int)
  y_pred_iris_class = np.clip(y_pred_iris_class, 0, 2)
  return  y_pred_iris_class

def plot_confusion_matrix(cm, class_names, title):
    """Render a confusion matrix as a labeled heatmap.
      args:
        cm: Confusion matrix (2D array) where rows are true classes and columns are predicted classes.
        class_names: Human-readable names for each class index.
        title: Title to display above the heatmap.
      returns:
        None
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prediction')
    plt.ylabel('Real Value')
    plt.title(title)
    plt.show()
    
def function_show_graphics(y_test, y_pred_iris, y_pred_iris_class, test:int, scaler:str, target_names, plt_type):
  """Visualize model results before and after converting predictions to classes.
    args:
      y_test: True labels from the test split.
      y_pred_iris: Continuous predictions from LinearRegression.
      y_pred_iris_class: Discrete class predictions after rounding/clipping.
      scaler: Name of the scaler used ('Standard' or 'MinMax').
      target_names: Class display names for the confusion matrix axes.
      plt_type: Either 'Before' (scatter and residual plots) or 'After' (confusion matrix).
    returns:
      None
  """
  if plt_type == 'Before':
    try:
      #Scatter plot
      fig, axes = plt.subplots(1, 2, figsize=(16, 7))
      sns.regplot(x=y_test, y=y_pred_iris, ci=95, scatter_kws={'alpha':0.4, 'color': 'purple'}, ax=axes[0])
      axes[0].set_title('Actual Values vs. Predicted Values', fontsize=14)
      axes[0].set_xlabel("Actual Values")
      axes[0].set_ylabel("Predicted Values")


      #Residue graph
      sns.residplot(x=y_pred_iris, y=y_test, lowess=True, ax=axes[1], scatter_kws={'color': 'orange'})
      axes[1].set_title('Residuals Plot', fontsize=14)
      axes[1].set_xlabel("Predicted Values")
      axes[1].set_ylabel("Residuals (Errors)")

      fig.suptitle('Linear Regression Analysis in the Iris Dataset', fontsize=18)

      # Adjust the layout so that titles do not overlap
      plt.tight_layout(rect=[0, 0.03, 1, 0.95])

      # Display the graphics
      plt.show()
    except Exception as e:
      print(f"Error showing before class...: {e}")
  elif plt_type == 'After':
    try:
      cm = confusion_matrix(y_test, y_pred_iris_class)
      plot_confusion_matrix(cm, target_names, f"Confusion Matrix for test {test} using {scaler}")
    except Exception as e:
      print(f"Error showing after class...: {e}")
      
def function_metrics(X, y, scaler, random_state, iris_class_names, test)->None:
  """Train/evaluate LinearRegression on an 80/20 split and report regression and derived classification metrics.
    args:
      X: Feature matrix.
      y: Target vector with class indices 0, 1, 2.
      scaler: Scaling strategy ('Standard' or 'MinMax').
      random_state: Seed used for the train/test split (also reported as the test identifier).
      iris_class_names: Class display names in the order of label indices.
    returns:
      None
  """
  #Division of the dataset
  X_train, X_test, y_train, y_test=function_split(X, y, random_state)

  # Standard normalization of the dataset
  if scaler == 'Standard':
    X_train_scaled, X_test_scaled = function_standard_scaler(X_train, X_test)
  elif scaler == 'MinMax':
    X_train_scaled, X_test_scaled = function_min_max_scaler(X_train, X_test)

  # Model training with normalized data
  model_iris=function_fit(X_train_scaled, y_train)

  # Model prediction
  y_pred_iris=function_predict(model_iris, X_test_scaled)
  y_pred_iris_class=regression_to_class(y_pred_iris)

  try:
    # Model evaluation
    print(f"===== Before classification in test {test} using {scaler}=====\n")
    iris_mse=mean_squared_error(y_test, y_pred_iris)
    iris_mae=mean_absolute_error(y_test, y_pred_iris)
    iris_r2=r2_score(y_test, y_pred_iris)
    print(f"\n\nMean Absolute Error: {iris_mae} \n\nMean Square Error: {iris_mse} \n\nRsquared: {iris_r2}")
    function_show_graphics(y_test, y_pred_iris, y_pred_iris_class, test, scaler, iris_class_names, 'Before')

  except Exception as e:
    print(f"Error calculating metrics before class...: {e}")

  try:
    print(f"\n===== After classification in test {test} using {scaler} =====\n")
    iris_mse_class=mean_squared_error(y_test, y_pred_iris_class)
    iris_mae_class=mean_absolute_error(y_test, y_pred_iris_class)
    iris_r2_class=r2_score(y_test, y_pred_iris_class)
    iris_f1_class=f1_score(y_test, y_pred_iris_class, average='weighted')
    iris_conf_matrix_class=confusion_matrix(y_test, y_pred_iris_class)
    print(f"\n\nMean Absolute Error: {iris_mae_class} \n\nMean Square Error: {iris_mse_class} \n\nRsquared: {iris_r2_class}" )
    print(f"\n\nConfusion Matrix:\n {iris_conf_matrix_class}")
    print(f"\n\nClassification Report:\n {classification_report(y_test, y_pred_iris_class, target_names=iris_class_names)}")
    function_show_graphics(y_test, y_pred_iris, y_pred_iris_class, test, scaler, iris_class_names, 'After')

  except Exception as e:
    print(f"Error calculating metrics after class...: {e}")
    
def evualuate_with_cross_validation(X, y, model, scaler, iris_target_names):
  """Evaluate a model using 5-fold cross-validation, then derive classification metrics from regression outputs.
    args:
      X: Feature matrix.
      y: Target vector with class indices 0, 1, 2.
      model: Regressor to evaluate (e.g., LinearRegression()).
      scaler: Scaling strategy ('Standard' or 'MinMax') applied before CV.
      iris_target_names: Class display names in label order.
    returns:
      None
  """
  try:
    if scaler == 'Standard':
      X_scaled = StandardScaler().fit_transform(X)
    elif scaler == 'MinMax':
      X_scaled = MinMaxScaler().fit_transform(X)

    scores = cross_val_predict(model, X_scaled, y, cv=5)
    scores_class = regression_to_class(scores)
    ac_score = accuracy_score(y, scores_class)
    cm = confusion_matrix(y, scores_class)
    report = classification_report(y, scores_class, target_names=iris_target_names)

    print(f"==== Model evaluation using K-fold-cross-validation using {scaler} ====\n")
    print(f"Accuracy: {ac_score:.2%}\n")
    print(f"Confusion matrix:\n{cm}\n")
    print(f"Classification report:\n{report}\n")
    plot_confusion_matrix(cm, iris_target_names, "Confusion Matrix")
    print("\n\n")
  except Exception as e:
    print(f"Error during k-fold-cross-validation...: {e}")
    
def main():
  """Entry point to run the Iris Linear Regression workflow with CV and multiple train/test splits.
    args:
      None
    returns:
      None
  """
  # load the dataset into variable ds_iris
  ds_iris = load_iris()
  print(f"Target variables dataset:\n {ds_iris.target}\n\nTarget variable names: {ds_iris.target_names} \n\nFeature names: {ds_iris.feature_names}\n")

  # Creation of features X and target variable Y
  X = pd.DataFrame(ds_iris.data, columns=ds_iris.feature_names)
  y = ds_iris.target
  names = ds_iris.target_names

  # First analysis with k-fold cross-validation
  evualuate_with_cross_validation(X, y, LinearRegression(), 'Standard', names)
  evualuate_with_cross_validation(X, y, LinearRegression(), 'MinMax', names)

  # Tests with 80/20 training and different parameters
  print("==== Test using train split 80/20 ====")
  i = 0
  for rs in [42, 53, 9722, 17]:
    i += 1
    function_metrics(X, y, 'Standard', rs, names, i)
    function_metrics(X, y, 'MinMax', rs, names, i)

if __name__ == "__main__":
  main()
