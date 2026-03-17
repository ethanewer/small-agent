#!/bin/bash
set -e

containers=$(docker ps -q)

if [ -z "$containers" ]; then
    echo "No running containers."
    exit 0
fi

echo "$containers" | xargs docker stop
echo "Stopped $(echo "$containers" | wc -l | tr -d ' ') container(s)."
