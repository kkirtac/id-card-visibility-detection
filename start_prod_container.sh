#!/bin/bash

if [ $# -eq 0 ]; then

    echo "No container name provided"

    echo "Usage: $0 <containername>"

    exit 1

fi

CONTAINER_NAME=$1

echo "Start Building $CONTAINER_NAME ..." 

docker compose -f docker-compose.prod.yml -p $CONTAINER_NAME down && docker compose -f docker-compose.prod.yml -p $CONTAINER_NAME up --build

show_help() {
    echo -e "-b, --b" \
        "\n\t Rebuild the docker images" \
        "\n-h, --help" \
        "\n\t Display this help and exit"
}
