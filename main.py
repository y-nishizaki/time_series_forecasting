import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import mlflow
from mlflow.pyfunc import PythonModel
import seaborn as sns
import statistics

class ProphetWrapper(PythonModel):
    def __init__(self, model: Optional[Prophet] = None) -> None:
        """
        Initialize the ProphetWrapper class.

        Parameters
        ----------
        model : Optional[Prophet]
            Pretrained Prophet model, default is None.
        """
        self.model = model if model else Prophet()

    def fit(self, df: pd.DataFrame, external_features: List[str]) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the training data.
        external_features : List[str]
            List of external feature names to add as regressors.
        """
        for feature in external_features:
            self.model.add_regressor(feature)
        self.model.fit(df)

    def predict(self, context: None, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using the model.

        Parameters
        ----------
        context : None
            Not used.
        model_input : pd.DataFrame
            Dataframe containing features for prediction.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the predictions.
        """
        return self.model.predict(model_input)

def create_sample_data(
        n: int = 100, 
        start_date: str = '2020-01-01', 
        end_date: str = '2020-04-10', 
        keys1: List[str] = ['A', 'B'], 
        keys2: List[str] = ['X', 'Y']
    ) -> Tuple[pd.DataFrame, int]:
    """
    Create sample data.

    Parameters
    ----------
    n : int
        Number of data points.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    keys1 : List[str]
        List of keys for the 'KEY1' column.
    keys2 : List[str]
        List of keys for the 'KEY2' column.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        Generated sample dataframe and train size.
    """
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

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.

    Returns
    -------
    Dict[str, float]
        Dictionary containing the calculated metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    error_rate = abs((np.sum(y_true) - np.sum(y_pred)) / np.sum(y_true)) * 100
    return {'RMSE': rmse, 'MAE': mae, 'Error Rate': error_rate}

def log_metrics_and_artifacts(
        metrics: Dict[str, float], 
        prediction_plot_path: str, 
        components_plot_path: str,
        model: Prophet, 
        keys_dict: Dict[str, str], 
        experiment_id: str
    ) -> None:
    """
    Log metrics and artifacts.

    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metrics to log.
    prediction_plot_path : str
        Path to save the prediction plot.
    components_plot_path : str
        Path to save the components plot.
    model : Prophet
        Trained Prophet model.
    keys_dict : Dict[str, str]
        Dictionary containing keys.
    experiment_id : str
        ID of the mlflow experiment.
    """
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"Model_{keys_dict['KEY1']}_{keys_dict['KEY2']}", nested=True):
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.log_artifact(prediction_plot_path)
        mlflow.log_artifact(components_plot_path)
        wrapped_model = ProphetWrapper()
        wrapped_model.model = model
        mlflow.pyfunc.log_model("model", python_model=wrapped_model)

def train_and_evaluate(
        df: pd.DataFrame, 
        train_size: int, 
        keys_dict: Dict[str, str], 
        experiment_id: str
    ) -> Dict[str, float]:
    """
    Train and evaluate the model.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing data.
    train_size : int
        Size of the training set.
    keys_dict : Dict[str, str]
        Dictionary containing keys for subsetting.
    experiment_id : str
        ID of the mlflow experiment.

    Returns
    -------
    Dict[str, float]
        Dictionary containing evaluation metrics.
    """
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

    return metrics

def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics over multiple models.

    Parameters
    ----------
    metrics_list : List[Dict[str, float]]
        List of metrics dictionaries.

    Returns
    -------
    Dict[str, float]
        Aggregated metrics.
    """
    aggregated_metrics = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        aggregated_metrics[f"{key}_mean"] = statistics.mean(values)
        aggregated_metrics[f"{key}_median"] = statistics.median(values)
        aggregated_metrics[f"{key}_max"] = max(values)
        aggregated_metrics[f"{key}_min"] = min(values)
    return aggregated_metrics



def main() -> None:
    """
    Main function to orchestrate training, evaluation and logging.
    """
    start_date = '2020-01-01'
    end_date = '2020-04-10'
    df, train_size = create_sample_data(start_date=start_date, end_date=end_date)
    key_columns = ['KEY1', 'KEY2']
    if not all(key in df.columns for key in key_columns):
        raise ValueError(f"One or more specified key columns {key_columns} do not exist in the dataframe.")
    unique_keys = df[key_columns].drop_duplicates()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_name = 'My_Time_Series_Evaluation_With_Multiple_External_Features'
    # experiment_name = 'Default'
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    metrics_list = []
    with mlflow.start_run(experiment_id=str(experiment_id), run_name="Grouped_Models"):
        for _, row in unique_keys.iterrows():
            keys_dict = row.to_dict()
            metrics = train_and_evaluate(df, train_size, keys_dict, experiment_id)
            metrics_list.append(metrics)

        aggregated_metrics = aggregate_metrics(metrics_list)
        for metric_name, metric_value in aggregated_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        df_metrics = pd.DataFrame(metrics_list)
        for metric in df_metrics.columns:
            plt.figure()
            sns.histplot(df_metrics[metric], kde=True)
            plot_path = f"./tmp/{metric}_distribution.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)



if __name__ == "__main__":
    main()
