import absl
import os
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma
from typing import List


DATA_DIR = os.path.join(os.getcwd(), "../../data/titanic")

MAX_CATEGORICAL_FEATURE_VALUES = [
    3,
    2,
]
CATEGORICAL_FEATURE_KEYS = [
    "pclass",
    "is_alone",
]

DENSE_FLOAT_FEATURE_KEYS = ["fare", "family_size", "age_pclass", "name_count", "cabin_count", "fare_per_person"]

BUCKET_FEATURE_BUCKET_COUNT = [3, 10, 10]
BUCKET_FEATURE_KEYS = ["age", "sibsp", "parch"]

VOCAB_SIZE = 1000
OOV_SIZE = 10
VOCAB_FEATURE_KEYS = ["boat", "embarked", "sex", "home.dest", "ticket_type"]

LABEL_KEY = "survived"

EPOCHS = 10
LEARNING_RATE = 0.001
NUM_STEPS = 1000


METRICS_SPECS = [
    tfma.MetricsSpec(
        metrics=[
            tfma.MetricConfig(
                class_name="BinaryAccuracy",
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(lower_bound={"value": 0.9}),
                    # change_threshold=tfma.GenericChangeThreshold(direction=tfma.MetricDirection.HIGHER_IS_BETTER,absolute={'value': 0.01})
                ),
            ),
            tfma.MetricConfig(class_name="AUC"),
        ]
    )
]


SLICING_SPECS = [
    tfma.SlicingSpec(),
    # tfma.SlicingSpec(feature_keys=['sex'])
]


def transformed_name(key):
    return key + "_xf"


def transformed_names(keys):
    return [transformed_name(key) for key in keys]


def fill_in_missing(x, output_dim=1, is_ts=False):
    """
    Input: index+value pairs
    """
    if isinstance(x, tf.SparseTensor):
        # From TFX CsvGen it comes as Sparse
        default_value = "" if x.dtype == tf.string else 0
        result = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], output_dim]),
            default_value,
        )
    else:
        # From csv reader it comes as Dense already
        result = tf.reshape(x, [-1, output_dim])
    if not is_ts:
        # must be the leangth of the tensor and remove the last dimension: (10,1) -> (10)
        return tf.squeeze(result, axis=1)
    else:
        # do not squeeze: preserve (10,5). tf.squeeze fails for (10,5)
        return result


def tranform_tensors(inputs):
    # we must use tf or tft only
    # substr works with dense tensor only

    inputs["ticket_type"] = tf.strings.substr(fill_in_missing(inputs["ticket"]), 0, 3)

    inputs["family_size"] = fill_in_missing(inputs["sibsp"]) + fill_in_missing(inputs["parch"]) + 1

    inputs["is_alone"] = tf.cast(tf.math.equal(inputs["family_size"], tf.constant(1, dtype=tf.int64)), tf.int64)

    inputs["age_pclass"] = tf.math.multiply(
        tf.cast(fill_in_missing(inputs["age"]), tf.float32), tf.cast(fill_in_missing(inputs["pclass"]), tf.float32)
    )

    inputs["name_count"] = tft.word_count(tf.strings.split(fill_in_missing(inputs["name"]), sep=" ")) - 1

    inputs["cabin_count"] = tft.word_count(tf.strings.split(fill_in_missing(inputs["cabin"]), sep=" ")) - 1

    inputs["fare_per_person"] = tf.math.multiply(
        tf.cast(fill_in_missing(inputs["fare"]), tf.float32),
        tf.cast(fill_in_missing(inputs["family_size"]), tf.float32),
    )

    outputs = {}

    for key in DENSE_FLOAT_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_z_score(fill_in_missing(inputs[key]))

    for key in VOCAB_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(
            fill_in_missing(inputs[key]), top_k=VOCAB_SIZE, num_oov_buckets=OOV_SIZE
        )

    for key, num_buckets in zip(BUCKET_FEATURE_KEYS, BUCKET_FEATURE_BUCKET_COUNT):
        outputs[transformed_name(key)] = tft.bucketize(fill_in_missing(inputs[key]), num_buckets)

    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[transformed_name(key)] = fill_in_missing(inputs[key])

    outputs[transformed_name(LABEL_KEY)] = fill_in_missing(inputs[LABEL_KEY])

    return outputs


def build_keras_model(hidden_units: List[int] = None) -> tf.keras.Model:

    " MAIN FUNCTION FOR TFX. NO TFX IN THIS MODULE"
    real_valued_columns = [
        tf.feature_column.numeric_column(key, shape=()) for key in transformed_names(DENSE_FLOAT_FEATURE_KEYS)
    ]
    categorical_columns = [
        tf.feature_column.categorical_column_with_identity(key, num_buckets=VOCAB_SIZE + OOV_SIZE, default_value=0)
        for key in transformed_names(VOCAB_FEATURE_KEYS)
    ]
    categorical_columns += [
        tf.feature_column.categorical_column_with_identity(  # pylint: disable=g-complex-comprehension
            key, num_buckets=num_buckets, default_value=0
        )
        for key, num_buckets in zip(transformed_names(BUCKET_FEATURE_KEYS), BUCKET_FEATURE_BUCKET_COUNT)
    ]
    categorical_columns += [
        tf.feature_column.categorical_column_with_identity(  # pylint: disable=g-complex-comprehension
            key, num_buckets=num_buckets, default_value=0
        )
        for key, num_buckets in zip(transformed_names(CATEGORICAL_FEATURE_KEYS), MAX_CATEGORICAL_FEATURE_VALUES)
    ]

    indicator_column = [
        tf.feature_column.indicator_column(categorical_column) for categorical_column in categorical_columns
    ]

    model = wide_and_deep_classifier(
        wide_columns=indicator_column,
        deep_columns=real_valued_columns,
        dnn_hidden_units=hidden_units or [100, 70, 50, 25],
    )
    return model


def wide_and_deep_classifier(wide_columns, deep_columns, dnn_hidden_units):

    input_layers = {
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
        for colname in transformed_names(DENSE_FLOAT_FEATURE_KEYS)
    }
    input_layers.update(
        {
            colname: tf.keras.layers.Input(name=colname, shape=(), dtype="int32")
            for colname in transformed_names(VOCAB_FEATURE_KEYS)
        }
    )
    input_layers.update(
        {
            colname: tf.keras.layers.Input(name=colname, shape=(), dtype="int32")
            for colname in transformed_names(BUCKET_FEATURE_KEYS)
        }
    )
    input_layers.update(
        {
            colname: tf.keras.layers.Input(name=colname, shape=(), dtype="int32")
            for colname in transformed_names(CATEGORICAL_FEATURE_KEYS)
        }
    )

    deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
    for numnodes in dnn_hidden_units:
        deep = tf.keras.layers.Dense(numnodes)(deep)
    wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(tf.keras.layers.concatenate([deep, wide]))

    output = tf.squeeze(output, -1)

    model = tf.keras.Model(input_layers, output)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
    )
    model.summary(print_fn=absl.logging.info)
    return model


if __name__ == "__main__":

    import tempfile
    import pandas as pd
    import tensorflow_data_validation as tfdv
    import tensorflow_transform.beam.impl as tft_beam
    from tensorflow_metadata.proto.v0 import schema_pb2

    path = '/Users/rkazinnik/PycharmProjects/jim/azumo/kazinnik_mlops'
    data_file = f"{path}/data/titanic/titanic.csv"

    # Schema
    print("*********************Schema*********************")
    feature_spec = {colname: tf.io.FixedLenFeature([], tf.string) for colname in VOCAB_FEATURE_KEYS}
    stats = tfdv.generate_statistics_from_csv(data_file)
    schema = tfdv.infer_schema(statistics=stats)
    data_metadata = tft.tf_metadata.dataset_metadata.DatasetMetadata(schema)

    schema_dict = {}
    for feature in schema.feature:
        schema_dict[feature.name] = feature.type

    tfdv.display_schema(schema)

    print("\n\n\n")

    # Data
    print("*********************Dataset*********************")
    dataset = pd.read_csv(data_file)

    for key in schema_dict:
        if schema_dict[key] == schema_pb2.FeatureType.INT:
            dataset[key] = dataset[key].fillna(0).astype("int64")
        elif schema_dict[key] == schema_pb2.FeatureType.FLOAT:
            dataset[key] = dataset[key].fillna(0.0).astype("float32")
        elif schema_dict[key] == schema_pb2.FeatureType.BYTES:
            dataset[key] = dataset[key].fillna("").astype("bytes")

    dict_features = list(dataset.applymap(lambda x: [x]).to_dict("index").values())

    print(dataset.head())

    print("\n\n\n")

    print("*********************TFT Transform*********************")
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
        transformed_dataset, transform_fn = (
            dict_features,
            data_metadata,
        ) | tft_beam.AnalyzeAndTransformDataset(tranform_tensors)
    transformed_data, transformed_metadata = transformed_dataset
    print("\n\n\n")

    print("*********************Transformed Data*********************")
    df = pd.DataFrame(transformed_data)
    print(df.head())
    print("\n\n\n")

    print("*********************TF Dataset*********************")
    # Convert to pandas dataframe in order to be able to split into target and features
    target = df.pop(transformed_name(LABEL_KEY))
    train_dataset = tf.data.Dataset.from_tensor_slices((df.to_dict("list"), target.values))
    train_dataset = train_dataset.batch(1)
    for feat, targ in train_dataset.take(5):
        print("Features: {}, Target: {}".format(feat, targ))
    print("\n\n\n")

    print("*********************Build Keras Model*********************")
    model = build_keras_model()
    print("\n\n\n")

    print("*********************Model Summary*********************")
    model.summary()
    print("\n\n\n")

    print("*********************Model Fit*********************")
    model.fit(train_dataset, epochs=EPOCHS)
    print("\n\n\n")

    print("*********************Model Predict*********************")
    val_dataset = tf.data.Dataset.from_tensor_slices((df.head(10).to_dict("list")))
    val_dataset = val_dataset.batch(1)
    a = model.predict(val_dataset)
    print(a)
