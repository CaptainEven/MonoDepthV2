python3 ./train.py \
        --num_workers 15 \
        --batch_size 15 \
        --num_epochs 10 \
        --load_weights_folder ./log_kitti/stereo_model/models/weights_5/ \
        --learning_rate 5e-5