#!/usr/bin/env bash

project_root=$(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)
cd $project_root
docker buildx build . \
    -f beeai-agent/Dockerfile \
    -t gpt-researcher-agent:latest \
    --build-arg MANIFEST_B64=$(cat beeai-agent/agent.yaml | base64)
