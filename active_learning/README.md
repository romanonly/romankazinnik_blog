Text Deduplication with Active Learning 

1M input rows:
css_public_all_ofos_locations.csv

Dropbox location:
https://www.dropbox.com/s/kkyvdam20htcur9/css_public_all_ofos_locations.csv?dl=0

# Install

chmod +x sh_test.sh
chmod +x create_docker.sh 

## Clone, create docker image (10-15mins) and run four steps:

## Fast test: 10mins 20K records. Run all with 'all' (500K+ records, 10 hours)  

step_1_create_blocks.sh # fast test run. add 'all' full run 

step_2_train_clf.sh

step_3_dedup_run_clf.sh

step_4_final.sh


# Results

.\metrics

dedupped rows:
css_clean.csv

duplicate rows:
css_duplicates.csv

duplicate pairs:
css_duplicates_pairs.csv

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


# Four steps explained 

### Create blocks: blocks.pickle

### Manual Label and create classifier from container: clf.pickle

### Run inferences parallel by 100-blocks per process: deduplicate_big_{pool_id}.csv

### 10+ interactions to select probability threshold and create two outputs: 

css_clear.csv

css_duplicates.csv


# Algorithm

### 20 mins: Build docker image

### 10mins: Blocks
Create 32K record pair blocks

### 15mins: Initialize 
Create initial Labels from 2K blocks and initial Classifier

### 10mins: Iterate till Stopping Criteria:

Sampling Strategy: Pull from Unlabeled and Label manually

Retrain Classifier with newly added Labels

### 100mins = X32 cores 32K blocks at 10 blocks per min processing

Run Classifier Inferences for All Block of Pairs
