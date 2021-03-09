import os
from typing import List, Text
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options

import module


def get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(module.LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def input_fn(
    file_pattern: List[Text],
    data_accessor: DataAccessor,
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int = 200,
) -> tf.data.Dataset:
    return data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=module.transformed_name(module.LABEL_KEY)
        ),
        tf_transform_output.transformed_metadata.schema,
    )


# TFX Transform will call this function.
def preprocessing_fn(inputs):
    return module.tranform_tensors(inputs)


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, fn_args.data_accessor, tf_transform_output, 40)
    eval_dataset = input_fn(fn_args.eval_files, fn_args.data_accessor, tf_transform_output, 40)

    model = module.build_keras_model()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, update_freq="batch")

    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(fn_args.model_run_dir, "metrics_train.csv"), separator=",", append=False
    )

    model.fit(
        train_dataset,
        epochs=module.EPOCHS,
        steps_per_epoch=module.NUM_STEPS,
        validation_data=eval_dataset,
        validation_steps=module.NUM_STEPS,
        callbacks=[tensorboard_callback, csv_logger],
    )

    signatures = {
        "serving_default": get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
