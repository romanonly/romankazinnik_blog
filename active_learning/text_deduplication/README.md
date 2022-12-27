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

### 10mins: Create 32K Blocks of Pairs

### 15mins: Create Initial Labels from 2K blocks and initial Classifier

### 10mins: Iterate till Stopping Criteria:

Sampling Strategy: Pull from Unlabeled and Label manually

Retrain Classifier with newly added Labels

### 100mins = X32 cores 32K blocks at 10 blocks per min processing

Run Classifier Inferences for All Block of Pairs
