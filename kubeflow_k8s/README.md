# Install 

## Kubernetes and Kubeflow

install docker https://docs.docker.com/get-docker/

install kubectl https://kubernetes.io/docs/tasks/tools/

install kind https://kind.sigs.k8s.io/

MacOS example: brew install kind

install miniconda https://docs.conda.io/en/latest/miniconda.html

# Deploy 

Installing recommended local Kubernetes cluster v1.19.1 fails but v1.21.1 will do the work.

#### kind create cluster --image=kindest/node:v1.21.1

#### kubectl version  --output=yaml

Install Kubeflow

#### export PIPELINE_VERSION=1.7.0
#### kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION&timeout=300"
#### kubectl wait --for condition=established --timeout=300s crd/applications.app.k8s.io
#### kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION&timeout=300"

### kubectl get deploy -n kubeflow 
You should see all deployments with the READY status before you can proceed to the next section.

#### kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
#### http://localhost:8080

## Install two conda environments, create Kubeflow pipelines and run validation trainings in conda 

Kubeflow 1.7.0 requires Python >=3.6.1

conda create --name py39kfp170 python=3.9

conda activate py39kfp170

which pip   

pip3  install --upgrade kfp==1.7.0

### Create three Kubeflow pipelines:

python3 kfp_create_pipeline_two_components.py 

python3 kfp_create_pipeline_five_components.py

python3 kfp_create_pipeline_five_components.py 

# Results

Submit pipelines yaml and run default parameters: Kubeflow-Pipelines-Upload Pipeline

Pipelines:

pipeline_amazon_py.yaml

pipeline_five_components_py.yaml

pipeline_two_components_py.yaml


### When done you can delete Kubeflow and cluster
export PIPELINE_VERSION=1.7.0

kubectl delete -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"

kubectl delete -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"

kind delete cluster

## You can run local trainings in conda (no Kubeflow/k8s) 

conda create --name py37tf230 python=3.7

conda activate py37tf230

which pip3

pip3  install -r requirements_pip3.txt

pip3 install scikit-learn==0.21

python3 train_amazon.py

python3 train_amazon_refactured.py

python3 train.py


# References

Kubeflow/Kubernetes

https://colab.research.google.com/github/https-deeplearning-ai/machine-learning-engineering-for-production-public/blob/main/course4/week3-ungraded-labs/C4_W3_Lab_1_Intro_to_KFP/C4_W3_Lab_1_Kubeflow_Pipelines.ipynb#scrollTo=9bs8p5KZGCgI/C4_W3_Lab_1_Kubeflow_Pipelines.ipynb

Tensorflow examples:

https://www.coursera.org/learn/tensorflow-serving-docker-model-deployment/resources/TfTVb

https://github.com/snehankekre/Deploy-Deep-Learning-Models-TF-Serving-Docker



