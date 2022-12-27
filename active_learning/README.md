Text Deduplication with Active Learning 

1M input rows:
css_public_all_ofos_locations.csv

Dropbox location:
https://www.dropbox.com/s/kkyvdam20htcur9/css_public_all_ofos_locations.csv?dl=0

# Install

chmod +x sh_test.sh
chmod +x create_docker.sh 

# Run quck test (10mins 20K records): uncomment 'docker run' in create_docker.sh 
./create_docker.sh

# Run all (500K+ records, 10 hours)  
./create_docker.sh all

# Results: .\metrics

dedupped rows:
css_clean.csv

duplicate rows:
css_duplicates.csv

duplicate pairs:
css_duplicates_pairs.csv

