# Import the modules you will use
import kfp
# For creating the pipeline
from kfp.v2 import dsl
# Type annotations for the component artifacts
from kfp.v2.dsl import (Input, Output, Artifact, Dataset, Model, Metrics)
# For building components
from kfp.v2.dsl import component


@component(packages_to_install = ["pandas", "openpyxl"], output_component_file = "component_download_data.yaml")
def download_data(url: str, num_samples: int, output_csv: Output[Dataset]):
    import pandas as pd

    # Use pandas excel reader
    df = pd.read_excel(url)
    df = df.sample(frac = 1).reset_index(drop = True)
    if num_samples > 0:
        df = df.sample(n = min(len(df), num_samples))
    df.to_csv(output_csv.path, index = False)


@component(packages_to_install = ["pandas", "scikit-learn"], output_component_file = "component_split_data.yaml")
def split_data(input_csv: Input[Dataset], train_csv: Output[Dataset], test_csv: Output[Dataset]):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_csv.path)
    train, test = train_test_split(df, test_size = 0.2)

    train.to_csv(train_csv.path, index = False)
    test.to_csv(test_csv.path, index = False)


@component(packages_to_install = ["pandas", "numpy"], output_component_file = "component_preprocess_data.yaml")
def preprocess_data(input_train_csv: Input[Dataset], input_test_csv: Input[Dataset], output_train_x: Output[Dataset],
                    output_test_x: Output[Dataset], output_train_y: Output[Artifact], output_test_y: Output[Artifact]):
    import pandas as pd
    import numpy as np
    import pickle

    def format_output(data):
        y1 = data.pop('Y1')
        y1 = np.array(y1)
        y2 = data.pop('Y2')
        y2 = np.array(y2)
        return y1, y2

    def norm(x, train_stats):
        return (x - train_stats['mean']) / train_stats['std']

    train = pd.read_csv(input_train_csv.path)
    test = pd.read_csv(input_test_csv.path)

    train_stats = train.describe()

    # Get Y1 and Y2 as the 2 outputs and format them as np arrays
    train_stats.pop('Y1')
    train_stats.pop('Y2')
    train_stats = train_stats.transpose()

    train_Y = format_output(train)
    with open(output_train_y.path, "wb") as file:
        pickle.dump(train_Y, file)

    test_Y = format_output(test)
    with open(output_test_y.path, "wb") as file:
        pickle.dump(test_Y, file)

    # Normalize the training and test data
    norm_train_X = norm(train, train_stats)
    norm_test_X = norm(test, train_stats)

    norm_train_X.to_csv(output_train_x.path, index = False)
    norm_test_X.to_csv(output_test_x.path, index = False)


@component(packages_to_install = ["tensorflow", "pandas"], output_component_file = "component_train_model.yaml")
def train_model(input_train_x: Input[Dataset], input_train_y: Input[Artifact], output_model: Output[Model],
                output_history: Output[Artifact]):
    import pandas as pd
    import tensorflow as tf
    import pickle

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input

    norm_train_X = pd.read_csv(input_train_x.path)

    with open(input_train_y.path, "rb") as file:
        train_Y = pickle.load(file)

    def model_builder(train_X):
        # Define model layers.
        input_layer = Input(shape = (len(train_X.columns),))
        first_dense = Dense(units = '128', activation = 'relu')(input_layer)
        second_dense = Dense(units = '128', activation = 'relu')(first_dense)

        # Y1 output will be fed directly from the second dense
        y1_output = Dense(units = '1', name = 'y1_output')(second_dense)
        third_dense = Dense(units = '64', activation = 'relu')(second_dense)

        # Y2 output will come via the third dense
        y2_output = Dense(units = '1', name = 'y2_output')(third_dense)

        # Define the model with the input layer and a list of output layers
        model = Model(inputs = input_layer, outputs = [y1_output, y2_output])

        print(model.summary())

        return model

    model = model_builder(norm_train_X)

    # Specify the optimizer, and compile the model with loss functions for both outputs
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)
    model.compile(
        optimizer = optimizer, loss = {'y1_output': 'mse', 'y2_output': 'mse'},
        metrics = {'y1_output': tf.keras.metrics.RootMeanSquaredError(),
                   'y2_output': tf.keras.metrics.RootMeanSquaredError()}
    )
    # Train the model for 500 epochs
    history = model.fit(norm_train_X, train_Y, epochs = 100, batch_size = 10)
    model.save(output_model.path)

    with open(output_history.path, "wb") as file:
        pickle.dump(history.history, file)


@component(packages_to_install = ["tensorflow", "pandas"], output_component_file = "component_eval_model.yaml")
def eval_model(input_model: Input[Model], input_history: Input[Artifact], input_test_x: Input[Dataset],
               input_test_y: Input[Artifact], output_metrics: Output[Dataset], MLPipeline_Metrics: Output[Metrics]):
    import pandas as pd
    import tensorflow as tf
    import pickle

    model = tf.keras.models.load_model(input_model.path)

    norm_test_X = pd.read_csv(input_test_x.path)

    with open(input_test_y.path, "rb") as file:
        test_Y = pickle.load(file)

    # Test the model and print loss and mse for both outputs
    loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x = norm_test_X, y = test_Y)
    print(
        "Loss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".format(
            loss, Y1_loss, Y1_rmse, Y2_loss,
            Y2_rmse
        )
    )

    metrics = pd.DataFrame(
        list(
            zip(
                ['loss', 'Y1_loss', 'Y2_loss', 'Y1_rmse', 'Y2_rmse'],
                [loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse]
            )
        ),
        columns = ['Name', 'val']
    )
    metrics.to_csv(output_metrics.path, index = False)
    MLPipeline_Metrics.log_metric("loss", loss)
    MLPipeline_Metrics.log_metric("Y1_loss", Y1_loss)
    MLPipeline_Metrics.log_metric("Y2_loss", Y2_loss)
    MLPipeline_Metrics.log_metric("Y1_rmse", Y1_rmse)
    MLPipeline_Metrics.log_metric("Y2_rmse", Y2_rmse)


# Define a pipeline and create a task from a component:
@dsl.pipeline(name = "pipeline-five-comps", )
def my_pipeline(url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
                num_samples: int = -1):
    download_data_task = download_data(url = url, num_samples = num_samples)

    split_data_task = split_data(input_csv = download_data_task.outputs['output_csv'])

    preprocess_data_task = preprocess_data(
        input_train_csv = split_data_task.outputs['train_csv'],
        input_test_csv = split_data_task.outputs['test_csv']
    )

    train_model_task = train_model(
        input_train_x = preprocess_data_task.outputs["output_train_x"],
        input_train_y = preprocess_data_task.outputs["output_train_y"]
    )

    eval_model_task = eval_model(
        input_model = train_model_task.outputs["output_model"],
        input_history = train_model_task.outputs["output_history"],
        input_test_x = preprocess_data_task.outputs["output_test_x"],
        input_test_y = preprocess_data_task.outputs["output_test_y"]
    )

    return


package_path = 'pipeline_five.yaml'

kfp.compiler.Compiler(mode = kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
    pipeline_func = my_pipeline,
    package_path = package_path
)


# Stress


def run_exp(pipeline_func, params, pipeline_filename, pipeline_package_path):
    EXPERIMENT_NAME = 'rk_tests'

    if pipeline_filename is not None:
        kfp.compiler.Compiler(mode = kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
            pipeline_func, pipeline_filename
        )
    else:
        assert pipeline_package_path is not None
        pipeline_filename = pipeline_package_path

    client = kfp.Client()

    experiment = client.create_experiment(EXPERIMENT_NAME)

    client.run_pipeline(
        experiment_id = experiment.id,
        job_name = pipeline_func.__name__ + "-" + pipeline_filename + '-run',
        pipeline_package_path = pipeline_filename,
        params = params
    )


params = {"url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
          'num_samples': -1}

# Stress serial jobs: uncomment
# for num_samples in range(100, 110):
#    params['num_samples'] = num_samples
#    run_exp(my_pipeline, params = params, pipeline_filename = None, pipeline_package_path = package_path)
