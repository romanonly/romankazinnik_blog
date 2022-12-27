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

echo  -e "\nStep-3: create duplicate pairs\n"

echo -e "Expected result for full records:\n"

echo -e "INFO:__main__:writing metrics/duplicates_big_789.csv"
echo -e "INFO:__main__:metrics/duplicates_big_789.csv done"
echo -e "INFO:__main__:multiprocessing Pool finished: 1088 processes \n\n"

docker run  -v ${PWD}/metrics:/usr/app/metrics  ${full_image_name} python3 -m metrics.run_dedup

