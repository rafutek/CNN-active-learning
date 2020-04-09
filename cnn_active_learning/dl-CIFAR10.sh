#!/bin/sh
DATA_DIR="./data"
UNZIP_NAME="cifar-10-batches-py"
ARCHIVE_NAME="cifar10.tar.gz"
DATASET_URL=https://www.cs.toronto.edu/\~kriz/cifar-10-python.tar.gz

# Create data directory if not present
[ ! -d "${DATA_DIR}" ] && mkdir "${DATA_DIR}"

# Download dataset archive and unzip it
if [ ! -d "${DATA_DIR}/${UNZIP_NAME}" ]; then
    if [ ! -f "${DATA_DIR}/${ARCHIVE_NAME}" ]; then
        echo "Downloading CIFAR10 dataset..."
        curl ${DATASET_URL} > "${DATA_DIR}/${ARCHIVE_NAME}"
    fi
    
    if [ -f "${DATA_DIR}/${ARCHIVE_NAME}" ]; then
        echo "Unzip..."
        tar -xzvf ${DATA_DIR}/${ARCHIVE_NAME} -C ${DATA_DIR}
        rm ${DATA_DIR}/${ARCHIVE_NAME}
    else echo "The archive is not present" && exit 1
    fi
else
    echo "CIFAR10 already downloaded"
fi


