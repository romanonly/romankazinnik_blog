#!/bin/bash -e
image_name=gcr.io/rk_image/test001
image_tag=v0.1
full_image_name=${image_name}:${image_tag}

cd "$(dirname "$0")"

docker build -t "${full_image_name}" .
echo docker run rk_image/test001:v0.1 -m metrics.run_main --parameters

for i in $( ls ); do
    echo for_item: $i
done

PWD=$( pwd -P )
echo PWD=${PWD}

echo -e "\nStep-4: Calibrate and create unique and duplicate record lists(use Step-3 duplicate pairs) \n"

echo -e "run interactive container (or python interpreter as: python3 -mmetrics.run_final)"
echo -e "run_final: EXIT when done\n\n"


echo -e "final step expected result for full records:"
echo -e "INFO:__main__:writing: metrics/css_clean.csv, metrics/css_duplicates.csv"
echo -e "num duplicates=200822"
echo -e "num clean=443027"

echo -e "\n\n Type: python3 -mmetrics.run_final \n\n"
docker run  -v ${PWD}/metrics:/usr/app/metrics -it ${full_image_name} /bin/bash
