#!/bin/bash

ENTITY=$(jq -r '.wandb.entity' config.json)
PROJECT=$(jq -r '.wandb.project' config.json)

wandb sweep --entity "${ENTITY}" --project "${PROJECT}" sweep-config.yaml 2>&1 | tee /tmp/capture.out 
SWEEP_ID=$(cat /tmp/capture.out | grep -e "wandb: Created sweep with ID:" | awk -F':' '{print $NF}' | awk '{$1=$1;print}')

wandb agent --count 20 "${ENTITY}/${PROJECT}/${SWEEP_ID}"

wandb sweep --stop "${ENTITY}/${PROJECT}/${SWEEP_ID}"

