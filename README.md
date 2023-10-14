# Prophet Time Series Forecasting with MLflow

## Overview
This Python codebase demonstrates the use of Prophet, a time-series forecasting model developed by Facebook, in conjunction with MLflow for logging and evaluation. It allows for the inclusion of external features in the model and evaluates the performance using metrics such as RMSE, MAE, and Error Rate.

## Requirements
- Python 3.x
- Prophet
- pandas
- NumPy
- Matplotlib
- scikit-learn
- MLflow
- seaborn

## Functions

### ProphetWrapper
A wrapper class around the Prophet model to facilitate easier fitting and predicting.

### create_sample_data
Generates sample time-series data with external features.

### calculate_metrics
Calculates evaluation metrics including RMSE, MAE, and Error Rate.

### log_metrics_and_artifacts
Logs metrics and artifacts such as plots to MLflow.

### train_and_evaluate
Trains and evaluates a Prophet model on the provided dataset and logs the metrics to MLflow.

### aggregate_metrics
Aggregates metrics over multiple runs for comprehensive analysis.

## Execution Steps
1. Set up a virtual environment and install required packages.
2. Execute `main()` to start the training and evaluation process.

# install command
```
conda install pandas numpy matplotlib scikit-learn seaborn
conda install -c conda-forge prophet mlflow
```

# create conda.yaml
```
conda env export -n time_series_forecasting > conda.yaml
```

# run
```
mlflow run . --experiment-name My_Time_Series_Evaluation_With_Multiple_External_Features
```