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

# alias PWD=$("pwd -P")
PWD=$( pwd -P )
echo PWD=${PWD}


echo  -e "\nInstructions:\n uncomment to run\n"

echo  -e "\nStep-1: param = ${1}\n"
echo docker run  -v ${PWD}/metrics:/usr/app/metrics  ${full_image_name} python3 -m metrics.run_blocks ${1}

echo  -e "\nStep-2: Intreactuive Labeling\n"
echo "run in container run_clf: EXIT when done"
docker run  -v ${PWD}/metrics:/usr/app/metrics -it ${full_image_name} /bin/bash
echo python3 -m metrics.run_clf

echo  -e "\nStep-3: create duplicate pairs\n"
echo docker run  -v ${PWD}/metrics:/usr/app/metrics  ${full_image_name} python3 -m metrics.run_dedup

echo  -e "\nStep-4: Calibrate and create two lists: unique and duplicate (use Step-3 duplicate pairs) \n"
echo "run in container run_final: EXIT when done"
echo docker run  -v ${PWD}/metrics:/usr/app/metrics -it ${full_image_name} /bin/bash
echo python3 -mmetrics.run_final
