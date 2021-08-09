Requirement:
Tensorflow 2.5, CUDA 11.2.1, and CuDNN 8.1

Get data here [https://drive.google.com/drive/folders/1kifkUfn2OBRWJ0HOv4jQGPdYHX9-TJDH?usp=sharing](https://drive.google.com/drive/folders/1kifkUfn2OBRWJ0HOv4jQGPdYHX9-TJDH?usp=sharing). Request alexvn.work@gmail.com

----------------
Install
```
conda create -n tf-api pip python=3.6
conda activate tf-api
pip install --ignore-installed --upgrade tensorflow==2.5.0
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
# Setup gpu
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
# From within TensorFlow/models/research/
protoc object_detection/protos/*.proto --python_out=.
# From within TensorFlow/models/research/
# NOTE: You MUST open a new Terminal for the changes in the environment variables to take effect.
for /f %i in ('dir /b object_detection\protos\*.proto') do protoc object_detection\protos\%i --python_out=.
```

COCO
```
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

API
```
# From within TensorFlow/models/research/
# DONE: cp object_detection/packages/tf2/setup.py .
python -m pip install .
# From within TensorFlow/models/research/
python object_detection/builders/model_builder_tf2_test.py
```


Work:
```
conda install pandas

cd Tensorflow\scripts\preprocessing

python generate_tfrecord.py -x ../../workspace/training_demo/images/train -l ../../workspace/training_demo/annotations/label_map.pbtxt -o ../../workspace/training_demo/annotations/train.record

python generate_tfrecord.py -x ../../workspace/training_demo/images/test -l ../../workspace/training_demo/annotations/label_map.pbtxt -o ../../workspace/training_demo/annotations/test.record

python generate_tfrecord.py -x ../../workspace/training_demo_2/images/train -l ../../workspace/training_demo_2/annotations/label_map.pbtxt -o ../../workspace/training_demo_2/annotations/train.record

python generate_tfrecord.py -x ../../workspace/training_demo_2/images/test -l ../../workspace/training_demo_2/annotations/label_map.pbtxt -o ../../workspace/training_demo_2/annotations/test.record
```

Next, rmb to copy and edit the `pipeline.config`

```
# from Tensorflow/workspace/training_demo
# For ssd_resnet50_v1_fpn
python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config

# For centernet_mobilenet
python model_main_tf2.py --model_dir=models/centernet_mobilenet --pipeline_config_path=models/centernet_mobilenet/pipeline.config

# For centernet_resnet101
python model_main_tf2.py --model_dir=models/centernet_resnet101 --pipeline_config_path=models/centernet_resnet101/pipeline.config

# For efficientdet d7
python model_main_tf2.py --model_dir=models/efficientdet_d7 --pipeline_config_path=models/efficientdet_d7/pipeline.config

# For ssd_mobilenet_v2_fpnlite
python model_main_tf2.py --model_dir=models/ssd_mobilenet_v2_fpnlite --pipeline_config_path=models/ssd_mobilenet_v2_fpnlite/pipeline.config

```

View tensorboard
```
# from training_demo
tensorboard --logdir=models/my_ssd_resnet50_v1_fpn
```

Export model
```
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_resnet50_v1_fpn\pipeline.config --trained_checkpoint_dir .\models\my_ssd_resnet50_v1_fpn\ --output_directory .\exported-models\my_model

python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\ssd_mobilenet/pipeline.config --trained_checkpoint_dir .\models/ssd_mobilenet --output_directory .\exported-models\ssd_mobilenet

python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\centernet_resnet101/pipeline.config --trained_checkpoint_dir .\models/centernet_resnet101 --output_directory .\exported-models\centernet_resnet101

python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\ssd_mobilenet_v2_fpnlite/pipeline.config --trained_checkpoint_dir .\models/ssd_mobilenet_v2_fpnlite --output_directory .\exported-models\ssd_mobilenet_v2_fpnlite

```
