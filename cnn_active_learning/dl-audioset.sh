#!/bin/sh
DATA_DIR="./data"
DATA_UNZIP_NAME="audioset"
ARCHIVE_DATA_NAME="audioset.tar.gz"
DATASET_URL=http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv

FEATURE_DIR="./features"
FEATURES_UNZIP_NAME="features"
ARCHIVE_FEATURES_NAME="features.tar.gz"
FEATURES_URL_US=storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz
FEATURES_URL_EU=storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz

if [ ! -d "${DATA_DIR}/${DATA_UNZIP_NAME}" ]; then
    if [ ! -f "${DATA_DIR}/${ARCHIVE_DATA_NAME}" ]; then
        echo "Downloading audioset dataset..."
        curl ${DATASET_URL} > "${DATA_DIR}/${ARCHIVE_DATA_NAME}"
    fi

    if [ -f "${DATA_DIR}/${ARCHIVE_DATA_NAME}" ]; then
        echo "Unzip..."
        tar -xzvf ${DATA_DIR}/${ARCHIVE_DATA_NAME} -C ${DATA_DIR}
        rm ${DATA_DIR}/${ARCHIVE_DATA_NAME}
    else echo "The archive is not present" && exit 1
    fi
else
    echo "audioset dataset already downloaded"
fi



if [ ! -d "${FEATURE_DIR}/${FEATURES_UNZIP_NAME}" ]; then

    if [ $1 == "EU" ]; then
        echo "Downloading audioset dataset..."
        curl ${FEATURES_URL_EU} > "${FEATURE_DIR}/${ARCHIVE_FEATURES_NAME}"
    fi

    if [ $1 == "US" ]; then
        echo "Downloading audioset dataset..."
        curl ${FEATURES_URL_US} > "${FEATURE_DIR}/${ARCHIVE_FEATURES_NAME}"
    fi



    if [ -f "${FEATURE_DIR}/${ARCHIVE_FEATURES_NAME}" ]; then
        echo "Unzip..."
        tar -xzvf ${FEATURE_DIR}/${ARCHIVE_FEATURES_NAME} -C ${FEATURE_DIR}
        rm ${FEATURE_DIR}/${ARCHIVE_FEATURES_NAME}
    else echo "The archive is not present" && exit 1
    fi
else
    echo "audioset features already downloaded"
fi





