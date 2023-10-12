import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import mlflow
from mlflow.pyfunc import PythonModel
import seaborn as sns

class ProphetWrapper(PythonModel):
    def __init__(self, model=None):
        self.model = model if model else Prophet()

    def fit(self, df, external_features):
        for feature in external_features:
            self.model.add_regressor(feature)
        self.model.fit(df)

    def predict(self, context, model_input):
        return self.model.predict(model_input)

def create_sample_data(n=100, start_date='2020-01-01', end_date='2020-04-10', keys1=['A', 'B'], keys2=['X', 'Y']):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    train_size = int(0.8 * len(date_range))
    df = pd.DataFrame({
        'ds': np.tile(date_range, len(keys1) * len(keys2)),
        'y': np.sin(np.tile(np.linspace(0, 10, len(date_range)), len(keys1) * len(keys2))) + np.random.normal(0, 0.1, len(date_range) * len(keys1) * len(keys2)),
        'external_feature1': np.tile(np.linspace(0, 5, len(date_range)), len(keys1) * len(keys2)),
        'external_feature2': np.tile(np.linspace(5, 10, len(date_range)), len(keys1) * len(keys2)),
        'KEY1': np.repeat(keys1, len(date_range) * len(keys2)),
        'KEY2': np.tile(np.repeat(keys2, len(date_range)), len(keys1))
    })
    return df, train_size

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    error_rate = abs((np.sum(y_true) - np.sum(y_pred)) / np.sum(y_true)) * 100
    return {'RMSE': rmse, 'MAE': mae, 'Error Rate': error_rate}

def log_metrics_and_artifacts(metrics, prediction_plot_path, components_plot_path, model, keys_dict, experiment_id):
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"Model_{keys_dict['KEY1']}_{keys_dict['KEY2']}", nested=True):
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.log_artifact(prediction_plot_path)
        mlflow.log_artifact(components_plot_path)
        wrapped_model = ProphetWrapper()
        wrapped_model.model = model
        mlflow.pyfunc.log_model("model", python_model=wrapped_model)

def train_and_evaluate(df, train_size, keys_dict, experiment_id):
    tmp_path = './tmp'
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)    

    wrapped_model = ProphetWrapper()
    query_str = ' & '.join([f"{k} == '{v}'" for k, v in keys_dict.items()])
    df_filtered = df.query(query_str)
    df_train = df_filtered.iloc[:train_size]
    df_test = df_filtered.iloc[train_size:]
    wrapped_model.fit(df_train, ['external_feature1', 'external_feature2'])
    future = wrapped_model.model.make_future_dataframe(periods=len(df_test))
    future['external_feature1'] = df['external_feature1'].median()
    future['external_feature2'] = df['external_feature2'].median()
    forecast = wrapped_model.predict(None, future)
    y_true = df_test['y'].values
    y_pred = forecast['yhat'][train_size:].values
    metrics = calculate_metrics(y_true, y_pred)
    fig1 = wrapped_model.model.plot(forecast)
    prediction_plot_path = f"{tmp_path}/prediction_plot_{keys_dict['KEY1']}_{keys_dict['KEY2']}.png"
    plt.savefig(prediction_plot_path)
    fig2 = wrapped_model.model.plot_components(forecast)
    components_plot_path = f"{tmp_path}/components_plot_{keys_dict['KEY1']}_{keys_dict['KEY2']}.png"
    plt.savefig(components_plot_path)
    log_metrics_and_artifacts(metrics, prediction_plot_path, components_plot_path, wrapped_model.model, keys_dict, experiment_id)

def main():
    start_date = '2020-01-01'
    end_date = '2020-04-10'
    df, train_size = create_sample_data(start_date=start_date, end_date=end_date)
    key_columns = ['KEY1', 'KEY2']
    if not all(key in df.columns for key in key_columns):
        raise ValueError(f"One or more specified key columns {key_columns} do not exist in the dataframe.")
    unique_keys = df[key_columns].drop_duplicates()
    experiment_id = mlflow.get_experiment_by_name("My_Time_Series_Evaluation_With_Multiple_External_Features").experiment_id
    with mlflow.start_run(experiment_id=str(experiment_id), run_name="Grouped_Models"):
        for _, row in unique_keys.iterrows():
            keys_dict = row.to_dict()
            train_and_evaluate(df, train_size, keys_dict, experiment_id)

if __name__ == "__main__":
    mlflow.set_experiment("My_Time_Series_Evaluation_With_Multiple_External_Features")
    main()
