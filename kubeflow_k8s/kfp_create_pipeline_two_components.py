# Import the modules you will use
import kfp
# For creating the pipeline
from kfp.v2 import dsl
# Type annotations for the component artifacts
from kfp.v2.dsl import (
    Input,
    Output,
    Dataset
)
# For building components
from kfp.v2.dsl import component


@component(
    packages_to_install = ["pandas", "openpyxl"],
    output_component_file = "component_download_data.yaml"
)
def download_data(url: str, num_samples: int, output_csv: Output[Dataset]):
    import pandas as pd

    # Use pandas excel reader
    df = pd.read_excel(url)
    df = df.sample(frac = 1).reset_index(drop = True)
    if num_samples > 0:
        df = df.sample(n = num_samples)
    df.to_csv(output_csv.path, index = False)


@component(
    packages_to_install = ["pandas", "scikit-learn"],
    output_component_file = "component_split_data.yaml"
)
def split_data(input_csv: Input[Dataset], train_csv: Output[Dataset], test_csv: Output[Dataset]):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_csv.path)
    train, test = train_test_split(df, test_size = 0.2)

    train.to_csv(train_csv.path, index = False)
    test.to_csv(test_csv.path, index = False)


@dsl.pipeline(
    name = "pipeline-two-comps",
)
def my_pipeline(
        url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        num_samples: int = -1
):
    download_data_task = download_data(url = url, num_samples = num_samples)
    split_data_task = split_data(input_csv = download_data_task.outputs['output_csv'])


kfp.compiler.Compiler(mode = kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
    pipeline_func = my_pipeline,
    package_path = 'pipeline_two_components_py.yaml')
