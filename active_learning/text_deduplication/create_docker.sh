#!/bin/bash -e
image_name=gcr.io/rk_image/test001
image_tag=v0.1
full_image_name=${image_name}:${image_tag}

cd "$(dirname "$0")"

docker build -t "${full_image_name}" .
ECHO docker run rk_image/test001:v0.1 -m metrics.run_main --parameters
alias PWD="pwd -P"

echo  -e "\nInstructions:\n"

echo  -e "\nStep-1:\n"
echo docker run  -v $(PWD)/metrics:/usr/app/metrics  ${full_image_name} python3 -m metrics.run_blocks

echo  -e "\nStep-2: Intreactuive Labeling\n"
echo docker run  -v $(PWD)/metrics:/usr/app/metrics -it ${full_image_name} /bin/bash
echo python3 -m metrics.run_clf

echo  -e "\nStep-3:\n"
echo docker run  -v $(PWD)/metrics:/usr/app/metrics  ${full_image_name} python3 -m metrics.run_dedup

echo  -e "\nStep-4: Intreactuive Labeling\n"
echo docker run  -v $(PWD)/metrics:/usr/app/metrics -it ${full_image_name} /bin/bash
echo python3 -mmetrics.run_final
