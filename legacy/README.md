# Legacy method of approaching the OCR-credit-card problem

* First, this use the sobel filter to detect the edge of the number
* It detects the bounding boxes of digits by using the expected box ratio of the digit
* After extracting each digit's bounding box, it use a pretrained conv model to predict each digit.
* The pretrained model is trained by the mnist image dataset and the [kaggle's credit cart separated digit dataset](https://www.kaggle.com/barbaravanaki/credit-card-number-images)

# Installation
1. Install Anaconda if you have not already.
2. Create conda environment from file:
    ```
    conda env create -f environment.yml
    ```
3. Activate the conda environment:
    ```
    conda activate ocr
    ```
4. Download the dataset: [this link](https://drive.google.com/drive/folders/1vHQyGucB1lTcixYLClwAkTEp7LkRnrYo?usp=sharing)
5. Dataset source:
  * [Credit card image generator](https://github.com/Ardesco/credit-card-generator): [Website](https://ardesco.keybase.pub/ccgenerator/)
  * [Kaggle's credit card whole image](https://drive.google.com/file/d/16AKBO51_VAD19Epu9zDaP-l87kK8WVDg/view?usp=sharing). [Kaggle Link](https://www.kaggle.com/leonardluo1998/credit-card-number-identification-system)
  * [Twitter's @need a debit card](https://twitter.com/needadebitcard?lang=en)
  * [Twitter's @creditcard publisher](https://twitter.com/cr3d1tc4rds?lang=en)
  * [Generator](https://herramientas-online.com/credit-card-generator-with-name.php)
  * [Generator](https://getcreditcardonline.com/custom-credit-card/)
  * [Kaggle's credit card separated image](https://drive.google.com/file/d/196piqGPep4kIr2jX-ps4ERgwehhp5WNE/view?usp=sharing). [Kaggle link](https://www.kaggle.com/barbaravanaki/credit-card-number-images)
  * [Le Duy Son's crawled images](https://drive.google.com/file/d/11UsyAbPtKDh5Q9ldQP9v8R9fMvaZsOIK/view?usp=sharing)
6. Labeling tools:
  * [Roboflow](https://app.roboflow.com/)
  * [Classic tool (Only for linux)](https://github.com/tzutalin/labelImg)
7. Add new dataset to the official dataset:
  * We can use the `dataset_adder.py` in the `dataset` folder:
  ```
  cd dataset
  python dataset_adder.py -s credit_card -t official_ds
  ```
  For more help, refer to `python dataset_adder.py --help`

# Project structure:

1. The `models` folder contains the saved models from training.
2. The folder `dataset` contains the exmaple credit card images.
3. [`main.py`](main.py) This file is the main notebook that predicts the credit card. As you can see in the file:
    * The file first load the pretrained model.
    * Perform finding contours and select the **region of interest** (in this case, group of numbers and digits).
    * Perform separating digit.
    * Use the pretrained model to predict each separating digit.
4. [`train_mnist_font.py`](train_mnist_font.py) This file contain the code to train the convolutional model on the `MNIST` hand-written digit dataset.
5. [`train_credit_font.py`](train_credit_font.py) This file contain the code to train the convolutional model on the fonts for credit cards dataset.
6. [`models.py`](models.py) The structure of models are also contained in this file. (The method `conv_model()` is also contained, this model can further be trained with more dataset in different fonts).
7. [`data.py`](data.py) Contains the datasets utilities (i.e, Tensorflow custom dataset,...)
8. The file [`utils.py`](utils.py) contains utilities.

# Slide:
[https://docs.google.com/presentation/d/1X--VHg9UoqE9hO9lRXUCnqtrCJb8wULnPZcpaQubbPs/edit?usp=sharing](https://docs.google.com/presentation/d/1X--VHg9UoqE9hO9lRXUCnqtrCJb8wULnPZcpaQubbPs/edit?usp=sharing)