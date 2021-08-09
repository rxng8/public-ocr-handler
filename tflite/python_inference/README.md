# install

```
conda create -n tflite-runtime pip python=3.6
pip install numpy pillow
pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime

# Infer
python inference_test.py -i ../../test.jpg -m ../exported-models/detect.tflite -l ./label.txt
```