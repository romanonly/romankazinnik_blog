# Import the modules you will use
import kfp

# For creating the pipeline
# Type annotations for the component artifacts
# For building components

package_path = 'pipeline_amazon.yaml'


def run_exp(job_index='-run', params=None, pipeline_filename=None, pipeline_package_path=None, pipeline_func=None):
    EXPERIMENT_NAME = 'stress_tests'

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
        job_name = pipeline_filename + job_index,
        pipeline_package_path = pipeline_filename,
        params = params
    )


params = {'epochs': 100, 'batch_size': 32, 'num_samples': -1, 'learning_rate': 1e-4}

# Stress serial jobs examples

# Run network components
initStress = True

if initStress:
    for learning_rate in [1e-1, 1e-2, 1e-3, 1e-4]:
        params['learning_rate'] = learning_rate
        run_exp(params = params, pipeline_package_path = package_path)
else:
    # stress test
    maxJobs = 20
    params['learning_rate'] = 1e-3
    arams['num_samples'] = 1000

    for ind, batch_size in enumerate(range(1, 1 + maxJobs)):
        params['batch_size'] = 8 * batch_size
        run_exp(job_index = f"-batch-index-{ind}", params = params, pipeline_package_path = package_path)
