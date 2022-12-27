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

