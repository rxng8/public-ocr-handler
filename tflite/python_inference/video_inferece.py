# %%

# based on https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/detect_picamera.py

from tensorflow.lite.python.interpreter import Interpreter, load_delegate
import argparse
import time
import cv2
import re
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import matplotlib.pyplot as plt


# %%

def draw_image(image, results, labels, size):
  result_size = len(results)
  for idx, obj in enumerate(results):
    print(obj)
    # Prepare image for drawing
    draw = ImageDraw.Draw(image)

    # Prepare boundary box
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * size[0])
    xmax = int(xmax * size[0])
    ymin = int(ymin * size[1])
    ymax = int(ymax * size[1])

    # Draw rectangle to desired thickness
    for x in range( 0, 4 ):
      draw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 255, 0))

    box = [xmin, ymin]

    # Annotate image with label and confidence score
    display_str = labels[obj['class_id']] + ": " + str(round(obj['score']*100, 2)) + "%"
    draw.text((box[0], box[1]), display_str)

    displayImage = np.asarray( image )
    %matplotlib inline
    plt.imshow(displayImage)
    plt.show()


def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  print(boxes)
  print(classes)
  print(scores)
  print(count)

  results = []
  for i in range(count):
    # guarantee to have 1 result
    # if scores[i] >= threshold:
    if scores >= threshold:
      result = {
        'bounding_box': boxes, #[i],
        'class_id': int(classes), #[i],
        'score': scores #[i]
      }
      results.append(result)
  return results


def make_interpreter(model_file, use_edgetpu):
  model_file, *device = model_file.split('@')
  if use_edgetpu:
    return Interpreter(
      model_path=model_file,
      experimental_delegates=[
        load_delegate('libedgetpu.so.1',
        {'device': device[0]} if device else {})
      ]
    )
  else:
    return Interpreter(model_path=model_file)


# %%

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--model', type=str, required=True, help='File path of .tflite file.')
parser.add_argument('-l', '--labels', type=str, required=True, help='File path of labels file.')
parser.add_argument('-t', '--threshold', type=float, default=0.4, required=False, help='Score threshold for detected objects.')
parser.add_argument('-p', '--picamera', action='store_true', default=False, help='Use PiCamera for image capture')
parser.add_argument('-e', '--use_edgetpu', action='store_true', default=False, help='Use EdgeTPU')
args = parser.parse_args(['-m', '../exported-models/lite-model_keras-ocr_float16_2.tflite', '-l', './label.txt'])


# %%

labels = load_labels(args.labels)
interpreter = make_interpreter(args.model, args.use_edgetpu)
interpreter.allocate_tensors()
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

# %%

out_details = interpreter.get_output_details()
out_details

# %%

# with pic

# IMG_PATH = "../../test_1.jpg"
IMG_PATH = "../../Tensorflow\\workspace\\number_detection\\images\\train\\1_jpg.rf.822383970516e60729930e060182e8a0.jpg"

image = Image.open(IMG_PATH)


# %%

image_pred = image.resize((input_width ,input_height), Image.ANTIALIAS)
# image_pred = ImageOps.grayscale(image_pred)
# Perform inference
results = detect_objects(interpreter, image_pred, args.threshold)

# %%



draw_image(image, results, labels, image.size)


# %%

DELAY_BETWEEN_INFERENCE = 200

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    
    # Process here
    # results = 0 

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw here on image!
    cv2.imshow('Result', image)

    if cv2.waitKey(33) == ord('q'):
      break
cap.release()
cv2.destroyWindow('Result')



