# Import the modules you will use
import kfp

# For creating the pipeline
# Type annotations for the component artifacts
# For building components

package_path = 'pipeline_amazon.yaml'


def run_exp(params=None, pipeline_filename=None, pipeline_package_path=None, pipeline_func=None):
    EXPERIMENT_NAME = 'stress_test'

    if pipeline_filename is not None:
        assert pipeline_func is not None
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
        job_name = pipeline_filename + '-run',
        pipeline_package_path = pipeline_filename,
        params = params
    )


params = {'epochs': 1000, 'batch_size': 12, 'num_samples': -1, 'learning_rate': 1e-4}

# Stress serial jobs examples

for batch_size in range(1, 100):
    params['batch_size'] = 8 * batch_size
    run_exp(params = params, pipeline_package_path = package_path)

for learning_rate in [1e-1, 1e-2, 1e-3, 1e-4]:
    params['learning_rate'] = learning_rate
    run_exp(params = params, pipeline_package_path = package_path)
