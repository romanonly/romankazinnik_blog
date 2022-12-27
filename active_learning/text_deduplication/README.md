# Reproduce

Create docker image (10-15mins):

./create_docker.sh



# Four steps: 

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

# Questions and future work

### Acuracy
How to estimate the actual classification accuracy attained by this algorithm?

### Experiment
Try new features and model architectures and improve the classification accuracy.

### Production-grade
current algorithms runs parallel and persists all computations but doesn't robust to failures. 
It needs to be restarted if even one block out of thousands fails to compute. How to implement an optimal reducer to aggregate all the results?

