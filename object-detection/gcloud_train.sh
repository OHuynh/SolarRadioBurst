BUCKET_NAME=burstproject_1
JOB_DIR="gs://$BUCKET_NAME/keras-job-dir"
python setup.py sdist
python slim/setup.py sdist

gsutil cp ssd_resnet50_gcloud.config gs://${BUCKET_NAME}/data/ssd_resnet50_gcloud.config

gcloud ml-engine jobs submit training solar_job_13 \
    --runtime-version 2.3 \
    --job-dir=gs://${BUCKET_NAME}/model_dir \
    --packages dist/object_detection-0.1.tar.gz,dist/slim-0.1.tar.gz \
    --module-name object_detection.model_main_tf2 \
    --region europe-west1 \
    --config cloud.yml \
    --python-version 3.7 \
    -- \
    --model_dir=gs://${BUCKET_NAME}/model_dir \
    --pipeline_config_path=gs://${BUCKET_NAME}/data/ssd_resnet50_gcloud.config \
    --num_train_steps=70000