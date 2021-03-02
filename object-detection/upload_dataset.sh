BUCKET_NAME=burstproject_1
DATA_DIR="gs://$BUCKET_NAME/data"

gsutil cp ../data_proc/TFRecord/Type4/HardUpdated/1/dataset_type_4_train.tfrecord gs://${BUCKET_NAME}/data/dataset_type_4_train.tfrecord
gsutil cp ../data_proc/TFRecord/Type4/HardUpdated/1/dataset_type_4_test.tfrecord gs://${BUCKET_NAME}/data/dataset_type_4_test.tfrecord

#gsutil cp ../data_proc/dataset_type_2_train.tfrecord gs://${BUCKET_NAME}/data/dataset_type_2_train.tfrecord
#gsutil cp ../data_proc/dataset_type_2_test.tfrecord gs://${BUCKET_NAME}/data/dataset_type_2_test.tfrecord
