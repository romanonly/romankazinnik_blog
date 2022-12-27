#!/bin/bash -e
image_name=rk_image/test001
image_tag=v0.1
full_image_name=${image_name}:${image_tag}

cd "$(dirname "$0")"


ECHO "RUN INSTRUCTIONS"
alias PWD="pwd -P"
ECHO $(PWD)
ECHO  ${full_image_name}
ECHO docker run rk_image/test001:v0.1 -m metrics.run_dedup

echo -e "\nCheck copied files:\n"
ECHO -e "docker run -it python:3.6.10 /bin/bash"
ECHO docker run -it python:3.6.10 /bin/bash
ECHO docker run -it ${full_image_name}

echo  -e "\nCheck mapped files files ^d to exit:\n"
echo docker run image -v /absolute_path_local:/path_in_container
echo docker run  -v $(PWD)/metrics:/usr/app/metrics -it ${full_image_name} /bin/bash
echo ls metrics
echo python3 -m metrics.run_dedup
