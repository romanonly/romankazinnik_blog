# Import the modules you will use
import kfp
# For creating the pipeline
from kfp.v2 import dsl
# Type annotations for the component artifacts
from kfp.v2.dsl import (Input, Output, Artifact, Dataset)

# For building components
from kfp.v2.dsl import component


@component(
    packages_to_install = ["pandas", "openpyxl", "fsspec"],
    output_component_file = "component_amazon_load_dataset.yaml"
)
def load_dataset(url: str, num_samples: int, output_labels_artifacts: Output[Artifact],
                 output_text_artifacts: Output[Artifact]):
    import pandas as pd
    import pickle
    import numpy as np

    df = pd.read_csv(url, usecols = [6, 9])  # , nrows=num_samples)

    df.columns = ['rating', 'title']

    if num_samples > 1:
        df = df.sample(n = num_samples)

    text = df['title'].tolist()
    text = [str(t).encode('ascii', 'replace') for t in text]
    text = np.array(text, dtype = object)[:]

    labels = df['rating'].tolist()
    labels = [1 if i >= 4 else 0 if i == 3 else -1 for i in labels]
    labels = np.array(pd.get_dummies(labels), dtype = int)[:]

    with open(output_labels_artifacts.path, "wb") as file:
        pickle.dump(labels, file)

    with open(output_text_artifacts.path, "wb") as file:
        pickle.dump(text, file)


@component(
    packages_to_install = ["tensorflow", "tensorflow_hub"],
    output_component_file = "component_nnlm_model.yaml"
)
def nnlm_model_download(untrained_model: Output[Artifact]):
    import tensorflow as tf
    import tensorflow_hub as hub

    hub_layer = hub.KerasLayer(
        "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", output_shape = [128], input_shape = [],
        dtype = tf.string, name = 'input', trainable = False
    )

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(3, activation = 'softmax', name = 'output'))
    model.compile(
        loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy']
    )
    model.summary()
    print("\n\nsave untrained_model.pickle\n\n")
    model.save(untrained_model.path)


@component(
    packages_to_install = ["tensorflow", "pandas"],
    output_component_file = "component_train_model.yaml"
)
def train(epochs: int,
          batch_size: int,
          input_labels_artifacts: Input[Artifact],
          input_text_artifacts: Input[Artifact],
          input_untrained_model: Input[Artifact],
          output_model: Output[Artifact],
          output_history: Output[Artifact],
          output_metrics: Output[Dataset]
          ):
    import tensorflow as tf
    import pickle
    import pandas as pd

    print("Training the model ...")

    with open(input_labels_artifacts.path, "rb") as file:
        y_train = pickle.load(file)

    with open(input_text_artifacts.path, "rb") as file:
        x_train = pickle.load(file)

    # model = get_model()
    model = tf.keras.models.load_model(input_untrained_model.path)

    history = model.fit(
        x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_split = 0.2,
    )
    loss, acc = model.evaluate(x = x_train, y = y_train)
    print("\n\n\n Loss = {}, Acc = {} ".format(loss, acc))
    metrics = pd.DataFrame(
        list(
            zip(
                ['train_loss', 'train_accuracy'], [loss, acc]
            )
        ), columns = ['Name', 'val']
    )
    model.save(output_model.path)
    with open(output_history.path, "wb") as file:
        pickle.dump(history.history, file)
    metrics.to_csv(output_metrics.path, index = False)


# fails when explicitly install pickle


@component(
    packages_to_install = ["tensorflow", "pandas"], output_component_file = "component_eval_model.yaml"
)
def eval_model(input_model: Input[Artifact], input_labels_artifacts: Input[Artifact],
               input_text_artifacts: Input[Artifact], output_metrics: Output[Dataset]):
    import tensorflow as tf
    import pickle
    import pandas as pd

    with open(input_labels_artifacts.path, "rb") as file:
        y_eval = pickle.load(file)

    with open(input_text_artifacts.path, "rb") as file:
        x_eval = pickle.load(file)

    model = tf.keras.models.load_model(input_model.path)

    # Test the model and print loss and mse for both outputs
    loss, acc = model.evaluate(x = x_eval, y = y_eval)
    print("\n\n\n Loss = {}, Acc = {} ".format(loss, acc))
    metrics = pd.DataFrame(
        list(
            zip(
                ['loss', 'accuracy'], [loss, acc]
            )
        ), columns = ['Name', 'val']
    )
    metrics.to_csv(output_metrics.path, index = False)


@dsl.pipeline(name = "train_amazon_pipeline")
def my_pipeline(epochs: int = 10,
                batch_size: int = 12,
                num_samples: int = 10000,
                url_train: str = "https://www.dropbox.com/s/tdsek2g4jwfoy8q/train.csv?dl=1",
                url_test: str = "https://www.dropbox.com/s/tdsek2g4jwfoy8q/test.csv?dl=1"):
    download_train_data_task = load_dataset(url = url_train, num_samples = num_samples)

    nnlm_model = nnlm_model_download()

    train_model_task = train(
        epochs = epochs, batch_size = batch_size,
        input_labels_artifacts = download_train_data_task.outputs["output_labels_artifacts"],
        input_text_artifacts = download_train_data_task.outputs["output_text_artifacts"],
        input_untrained_model = nnlm_model.outputs["untrained_model"]
    )

    download_test_data_task = load_dataset(url = url_test, num_samples = num_samples)

    eval_model_task = eval_model(
        input_model = train_model_task.outputs["output_model"],
        input_labels_artifacts = download_test_data_task.outputs["output_labels_artifacts"],
        input_text_artifacts = download_test_data_task.outputs["output_text_artifacts"]
    )


kfp.compiler.Compiler(mode = kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
    pipeline_func = my_pipeline, package_path = 'pipeline_amazon.yaml'
)
