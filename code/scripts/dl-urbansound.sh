#!/bin/sh
DATA_DIR="./data"
UNZIP_NAME="UrbanSound"
ARCHIVE_NAME="UrbanSound.tar.gz"
DATASET_URL=https://goo.gl/8hY5ER

# Create data directory if not present
[ ! -d "${DATA_DIR}" ] && mkdir "${DATA_DIR}"

# Download dataset archive and unzip it
if [ ! -d "${DATA_DIR}/${UNZIP_NAME}" ]; then
    if [ ! -f "${DATA_DIR}/${ARCHIVE_NAME}" ]; then
        echo "Downloading UrbanSound dataset..."
        curl -L ${DATASET_URL} > "${DATA_DIR}/${ARCHIVE_NAME}"
    fi

    if [ -f "${DATA_DIR}/${ARCHIVE_NAME}" ]; then
        echo "Unzip..."
        tar -xzvf ${DATA_DIR}/${ARCHIVE_NAME} -C ${DATA_DIR} &&
            rm ${DATA_DIR}/${ARCHIVE_NAME}
    else echo "The archive is not present" && exit 1
    fi
else
    echo "UrbanSound already downloaded"
fi


