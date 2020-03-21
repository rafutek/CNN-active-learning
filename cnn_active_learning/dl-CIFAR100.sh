#!/bin/sh
DATA_DIR="./data"
UNZIP_NAME="cifar-100-python"
ARCHIVE_NAME="cifar100.tar.gz"
DATASET_URL=https://www.cs.toronto.edu/\~kriz/cifar-100-python.tar.gz

if [ ! -d "${DATA_DIR}/${UNZIP_NAME}" ]; then
    if [ ! -f "${DATA_DIR}/${ARCHIVE_NAME}" ]; then
        echo "Downloading CIFAR100 dataset..."
        curl ${DATASET_URL} > "${DATA_DIR}/${ARCHIVE_NAME}"
    fi

    if [ -f "${DATA_DIR}/${ARCHIVE_NAME}" ]; then
        echo "Unzip..."
        tar -xzvf ${DATA_DIR}/${ARCHIVE_NAME} -C ${DATA_DIR}
        rm ${DATA_DIR}/${ARCHIVE_NAME}
    else echo "The archive is not present" && exit 1
    fi
else
    echo "CIFAR100 already downloaded"
fi


