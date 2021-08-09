# Convert a normal saved_model to tflite model (don't do this with tf ofject detection api)

```
# from root directory
python ./tflite/tflite_converter.py -s ./Tensorflow\workspace\training_demo\exported-models\my_ssd_resnet50_v1_fpn/saved_model/ -t ./tflite\exported-models/ssd_resnet50_v1_fpn.tflite
```

or

```
# from root directory
tflite_convert --saved_model_dir=./Tensorflow\workspace\training_demo\exported-models\my_ssd_resnet50_v1_fpn/saved_model/ --output_file=./tflite\exported-models/ssd_resnet50_v1_fpn.tflite
```

# Instead do THIS: Export tf lite model

1. Export graph saved model
```
# this one does not work
# From root directory
python ./tflite/export_tflite_ssd_graph.py --pipeline_config_path=./Tensorflow\workspace\training_demo\models\my_ssd_resnet50_v1_fpn\pipeline.config --trained_checkpoint_prefix=./Tensorflow\workspace\training_demo\models\my_ssd_resnet50_v1_fpn\ckpt-26 --output_directory=./tflite\exported-models\ssd_resnet50_v1_fpn --add_postprocessing_op=true --max_detections=100

# This one works
# from root
python ./tflite/export_tflite_graph_tf2.py --pipeline_config_path=./Tensorflow\workspace\training_demo\models\my_ssd_resnet50_v1_fpn\pipeline.config --trained_checkpoint_dir=./Tensorflow\workspace\training_demo\models\my_ssd_resnet50_v1_fpn\ --output_directory=./tflite\exported-models\ssd_resnet50_v1_fpn/ --max_detections=100

# centernet_resnet101
python ./tflite/export_tflite_graph_tf2.py --pipeline_config_path=./Tensorflow\workspace\training_demo\models\centernet_resnet101\pipeline.config --trained_checkpoint_dir=./Tensorflow\workspace\training_demo\models\centernet_resnet101\ --output_directory=./tflite\exported-models\centernet_resnet101/ --max_detections=100

# ssd_mobilenet_v2_fpnlite
python ./tflite/export_tflite_graph_tf2.py --pipeline_config_path=./Tensorflow\workspace\training_demo\models\ssd_mobilenet_v2_fpnlite\pipeline.config --trained_checkpoint_dir=./Tensorflow\workspace\training_demo\models\ssd_mobilenet_v2_fpnlite\ --output_directory=./tflite\exported-models\ssd_mobilenet_v2_fpnlite/ --max_detections=25

# Number detection
python ./tflite/export_tflite_graph_tf2.py --pipeline_config_path=./Tensorflow\workspace\number_detection\models\ssd_mobilenet_v2_fpnlite\pipeline.config --trained_checkpoint_dir=./Tensorflow\workspace\number_detection\models\ssd_mobilenet_v2_fpnlite\ --output_directory=./tflite\exported-models\ssd_mobilenet_v2_fpnlite/ --max_detections=1
```

2. Export tflite from graph

```
# From root
tflite_convert --saved_model_dir=./tflite\exported-models\ssd_resnet50_v1_fpn/saved_model --output_file=./tflite\exported-models\detect.tflite --input_shapes=1,640,640,1 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_values=128 --change_concat_input_ranges=false --allow_custom_ops
```

## or use `tflite_converter.py` (Recommended)

## Before doing metadata, remember to:

```
pip install tflite_support==0.2.0
```

## And then write the metadata for tflite model with the file `metadata_writer.py` in `tflite/metadata_writer.py`

3. Infer tflite python

```
From within tflite\python_inference
python inference_test.py -i ../../test.jpg -m ../exported-models/detect.tflite -l ./label.txt
```

infer python with video:
```
# From tflite/python_inference
python video_inference_test.py -m ../exported-models/detect_metadata.tflite -l ./label.txt
```

4. Infer on android

This is hard =))
Write metadata to the head of tflite model if needed (you actually need it!)

-------------

# Other ways: Training with efficient det
# The folder doing this is `tflite/workspace`

1. download data and preprocess with `scripts/preprocess_csv.py`
2. train with `train_model.py`