# Public repository for optimizing OCR credit card reading problem with traditional methods and deep neural network methods.
# Author: Alex Nguyen (Viet Dung Nguyen)
## Demo
<center>
<p>
<img src="doc/demo1.gif" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="doc/demo2.gif" width="150">
</p>
</center>

## Algorithm List:
1. Legacy <`To be edited>`
2. CRNN <`To be edited>`
3. SSD <`To be edited>`
4. Efficientdet <`To be edited>`
5. Other: Faster-RCNN <`To be edited>`

## Prerequisite:
* Anaconda and related environment knowledge.
* VS Code (All files in this repo is programmed according to VS Code's jupyter notebook cell, `# %%`)
* A lot of work command-line interface, beware!
* Window Operating System: All the environment files are exported using a Window machine, so it will use some Window's sepcific library that Mac and Linux does not have! But if you choose not to use those environment files, then use whichever operating system you like!

## Dataset:
  * [Credit card image generator](https://github.com/Ardesco/credit-card-generator): [Website](https://ardesco.keybase.pub/ccgenerator/)
  * [Kaggle's credit card whole image](https://drive.google.com/file/d/16AKBO51_VAD19Epu9zDaP-l87kK8WVDg/view?usp=sharing). [Kaggle Link](https://www.kaggle.com/leonardluo1998/credit-card-number-identification-system)
  * [Twitter's @need a debit card](https://twitter.com/needadebitcard?lang=en)
  * [Twitter's @creditcard publisher](https://twitter.com/cr3d1tc4rds?lang=en)
  * [Generator](https://herramientas-online.com/credit-card-generator-with-name.php)
  * [Generator](https://getcreditcardonline.com/custom-credit-card/)
  * [Kaggle's credit card separated image](https://drive.google.com/file/d/196piqGPep4kIr2jX-ps4ERgwehhp5WNE/view?usp=sharing). [Kaggle link](https://www.kaggle.com/barbaravanaki/credit-card-number-images)
  * [Le Duy Son's crawled images](https://drive.google.com/file/d/11UsyAbPtKDh5Q9ldQP9v8R9fMvaZsOIK/view?usp=sharing)
  * Roboflow's Exported Dataset

## Labeling Tool
  * [Roboflow](https://app.roboflow.com/)
  * [Classic tool (Only for linux)](https://github.com/tzutalin/labelImg)

## Note
1. How to use conda environment files:
    * Edit the `name` field in the first line as you want.
    * Edit the `prefix` field in the last line according to this semantic: `<path-to-your-conda-envs-folder>/<name-of-the-environment>`. For example, path to your conda environment folder is: `C:\Users\nguyv\.conda\envs`, and your environment name is `ocr`, then the prefix's value should be: `C:\Users\nguyv\.conda\envs\ocr`
    * Create and activate the conda environment from the environment file, for example, `environment.yml`, with name, for example, `ocr`:
        ```
        conda env create -f environment.yml
        conda activate ocr
        ```

## Keyword:
1. Research from Duobango's solution.
   * gaussian blur (to get rid of the low level edge) ->'Sobel edge detector'...
   * 'Scharr edge detector'
   * 'Prewitt edge detector'
   * 'Canny edge detector'
   * 'Hough standard (STD)'
   * 'Kernel-based Hough transform (KHT)'
   * 'Standard Histogram of oriented gradients (S-HOG)'
   * 'Brute force matcher'
   * 'PLSL (Parallel Light Speed Labeling)'
   * 'LMSER (Linear Time Maximally Stable Extremal Regions)'

## Tensorflow Object detection API and Export TFLITE and TFJS.
## And also infer and test!

1. Navigate to [`Tensorflow\workspace\training_demo\README.md`](Tensorflow\workspace\training_demo\README.md)
  NOTE: Only ssd or center_net models are supported in tflite model conversion for tf object_detection library. (tf version: 2.5.0) (tf object detection version: 0.1)
2. Navigate to [`tflite\README.md`](tflite\README.md)

3. Stats:
  * Tensorflow models:
    * centernet_mobilenet (train success, have not inferred)
    * centernet_resnet101 (successfully inferred)
    * efficientdet_d7 (train failed)
    * my_ssd_resnet50_v1_fpn (successfully inferred)
    * ssd_mobilenet (failure inference)
    * ssd_mobilenet_v2_fpnlite (training)

4. Infer on mobile: follow this docs: [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md)

5. IMPORTANT: Tflite convert only support ssd now, according to [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md)

# Other ways on EfficientDet:
[https://www.tensorflow.org/lite/tutorials/model_maker_object_detection](https://www.tensorflow.org/lite/tutorials/model_maker_object_detection)
