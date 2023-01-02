# Results

css_clean.csv, css_duplicates.csv, css_duplicates_pairs.csv:
https://www.dropbox.com/scl/fo/6hu148v6yuwekgye6g7qb/h?dl=0&rlkey=c04k8nhdj82hhwpv257ap3hm3


# Questions and future work

### Acuracy
How to estimate the actual classification accuracy attained by this algorithm?

### Experiment
Try new features and model architectures and improve the classification accuracy.

### Compare
There are multiple deduplications packages. Run one of them and compare accuracies

### Production-grade
current algorithms runs parallel and persists all computations but doesn't robust to failures. 
It needs to be restarted if even one block out of thousands fails to compute. How to implement an optimal reducer to aggregate all the results?


# Reproduce

Clone, create docker image (10-15mins) and run four steps:

step_1_create_blocks.sh

step_2_train_clf.sh

step_3_dedup_run_clf.sh

step_4_final.sh

(also ./create_docker.sh)


# Four steps explained 

### Create blocks: blocks.pickle

### Manual Label and create classifier from container: clf.pickle

### Run inferences parallel by 100-blocks per process: deduplicate_big_{pool_id}.csv

### 10+ interactions to select probability threshold and create two outputs: 

### css_clear.csv

### css_duplicates.csv


# Algorithm

### 5 mins: Build docker image

### 10mins: Create record-pair blocks

### 15mins: Create initial labels from blocks and classifier

### 10mins: Iterate till Stopping Criteria:

Sampling Strategy: Pull from Unlabeled and Label manually

Retrain Classifier with newly added Labels
### 100mins = X32 cores 32K blocks at 10 blocks per min processing

Run Classifier Inferences for All Block of Pairs
