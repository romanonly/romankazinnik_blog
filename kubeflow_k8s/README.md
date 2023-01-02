# Blog

https://medium.com/how-to-turn-your-desktop-into-machine-learning/how-to-turn-research-code-into-production-grade-and-desktop-into-machine-learning-training-platform-da0ac9a8daf1

# Install 

## Kubernetes and Kubeflow

Install docker https://docs.docker.com/get-docker/

Install kubectl https://kubernetes.io/docs/tasks/tools/

Install kind https://kind.sigs.k8s.io/

MacOS example: brew install kind

~~Optional: install miniconda https://docs.conda.io/en/latest/miniconda.html~~

## Kubernetes 

Installing the recommended local (kind) Kubernetes cluster v1.19.1 is broken, use v1.21.1:

```
kind create cluster --image=kindest/node:v1.21.1
kubectl version  --output=yaml
```

## Kubeflow
```
export PIPELINE_VERSION=1.7.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION&timeout=300"
kubectl wait --for condition=established --timeout=300s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION&timeout=300"
kubectl get deploy -n kubeflow 
## You should see all deployments with the READY status before you can proceed to the next section.
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
http://localhost:8080
```

## Cluster dashboard (optional)

###  Dashboard UI 
https://kubernetes.io/docs/tasks/access-application-cluster/web-ui-dashboard/

>kubectl apply -f "https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml"

Create a token:
https://github.com/kubernetes/dashboard/blob/master/docs/user/access-control/creating-sample-user.md

>kubectl proxy 

Kubectl will make Dashboard available at: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/.

Select kubeflow cluster, Workloads, Deployments.

### Dashboard with Helm
https://medium.com/@munza/local-kubernetes-with-kind-helm-dashboard-41152e4b3b3d
https://helm.sh/docs/intro/install/
```
sudo dnf -y install arm-image-installer helm-3.5.4-2.fc35.x86_64
helm repo add kubernetes-dashboard https://kubernetes.github.io/dashboard/
helm install dashboard kubernetes-dashboard/kubernetes-dashboard -n kubernetes-dashboard --create-namespace
cat > service-account.yaml
kubectl apply -f service-account.yaml
kubectl describe serviceaccount admin-user -n kubernetes-dashboard
kubectl describe secret admin-user-token-xj8r5 -n kubernetes-dashboard
```
## Conda environments
### Install Kubeflow and Tensorflow conda environments
Kubeflow 1.7.0 requires Python >=3.6.1:
```
conda create --name py39kfp170 python=3.9
conda activate py39kfp170
pip3  install --upgrade kfp==1.7.0
```
### Create Kubeflow pipelines 
```
python3 kfp_train_amazon.py 
python3 kfp_create_pipeline_two_components.py
python3 kfp_create_pipeline_five_components.py
```

### Validate training 
```
conda create --name py37tf230 python=3.7
conda activate py37tf230
pip3  install -r requirements_pip3.txt
pip3 install scikit-learn==0.21
python3 train_amazon.py
python3 train_amazon_refactured.py
```

### Optional
Learn model training in **train.py** and **kfp_create_pipeline_two_components.py**. 

# Results

Upload pipelines yaml to cluster and run default parameters: **Kubeflow->Pipelines->Upload Pipeline**.

Pipelines: pipeline_amazon_py.yaml, pipeline_five_components_py.yaml, pipeline_two_components_py.yaml


### Delete Kubeflow and cluster
```
export PIPELINE_VERSION=1.7.0
kubectl delete -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"
kubectl delete -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kind delete cluster
```

# References

Kubeflow/Kubernetes

>https://colab.research.google.com/github/https-deeplearning-ai/machine-learning-engineering-for-production-public/blob/main/course4/week3-ungraded-labs/C4_W3_Lab_1_Intro_to_KFP/C4_W3_Lab_1_Kubeflow_Pipelines.ipynb#scrollTo=9bs8p5KZGCgI/C4_W3_Lab_1_Kubeflow_Pipelines.ipynb

Tensorflow model training examples:

>https://www.coursera.org/learn/tensorflow-serving-docker-model-deployment/resources/TfTVb

>https://github.com/snehankekre/Deploy-Deep-Learning-Models-TF-Serving-Docker



