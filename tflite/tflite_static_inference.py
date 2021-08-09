# %%

import numpy as np
import os
from tensorflow.python.ops.gen_math_ops import Imag

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
import matplotlib.pyplot as plt
import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# %%

import cv2

from PIL import Image

model_path = './workspace/model_det0_1.tflite'

# Load the labels into a list
classes = ['number']

# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  return resized_img, original_image


def set_input_tensor(interpreter, image):
  """Set the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Retur the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  # Feed the input image to the model
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all outputs from the model
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path,
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])

    # Find the class index of the current object
    class_id = int(obj['class_id'])

    # Draw the bounding box and label on the image
    color = [int(c) for c in COLORS[class_id]]
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    # label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
    label = "{}".format(classes[class_id])
    cv2.putText(original_image_np, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_image.numpy().astype(np.uint8), results, original_uint8

# %%

INPUT = "./test_images/test6.jpg"

DETECTION_THRESHOLD = 0.1

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
print(input_height)
print(input_width)

# Run inference and draw detection result on the local copy of the original file
original_img, results, detection_result_image = run_odt_and_draw_results(
  INPUT,
  interpreter,
  threshold=DETECTION_THRESHOLD
)

# Show the detection result
Image.fromarray(detection_result_image)

# plt.imshow(detection_result_image)
# plt.show()


# %%

cropped_image = np.zeros_like(original_img)

for obj in results:
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_img.shape[1])
    xmax = int(xmax * original_img.shape[1])
    ymin = int(ymin * original_img.shape[0])
    ymax = int(ymax * original_img.shape[0])
    cropped_image = original_img[ymin:ymax, xmin:xmax]
    break

# %%

model_path = './workspace_masked/model_det3_bs4_e150_2.tflite'

# Load the labels into a list
classes = ["0", "8", "2", "4", "6", "3", "1", "7", "9", "5"]

# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)
DETECTION_THRESHOLD = 0.3

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
print(input_height)
print(input_width)


# %%

# Test with real input

_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
# img = tf.io.read_file(REAL_IMG_PATH)
# img = tf.io.decode_image(img, channels=3)
img = tf.image.convert_image_dtype(cropped_image, tf.uint8)
# Pad to 1:1
img = tf.image.resize_with_pad(
  img,
  img.shape[1],
  img.shape[1],
  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
)
original_image = img

# Scale to input size
preprocessed_image = tf.image.resize(img, (input_height, input_width))
preprocessed_image = preprocessed_image[tf.newaxis, :]

# Run object detection on the input image
results = detect_objects(interpreter, preprocessed_image, threshold=DETECTION_THRESHOLD)

# Plot the detection results on the input image
original_image_np = original_image.numpy().astype(np.uint8)
for obj in results:
  # Convert the object bounding box from relative coordinates to absolute
  # coordinates based on the original image resolution
  ymin, xmin, ymax, xmax = obj['bounding_box']
  xmin = int(xmin * original_image_np.shape[1])
  xmax = int(xmax * original_image_np.shape[1])
  ymin = int(ymin * original_image_np.shape[0])
  ymax = int(ymax * original_image_np.shape[0])

  # Find the class index of the current object
  class_id = int(obj['class_id'])

  # Draw the bounding box and label on the image
  color = [int(c) for c in COLORS[class_id]]
  cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
  # Make adjustments to make the label visible for all objects
  y = ymin - 15 if ymin - 15 > 15 else ymin + 15
  label = "{}".format(classes[class_id])
  cv2.putText(original_image_np, label, (xmin, y),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

res_img = original_image_np.astype(np.uint8)
Image.fromarray(res_img)

# %%

# Print result

res_list = []
for obj in results:
  ymin, xmin, ymax, xmax = obj['bounding_box']
  xmin = int(xmin * original_img.shape[1])
  class_id = int(obj['class_id'])
  res_list.append((classes[class_id], xmin))

sorted_xmin_list = sorted(res_list, key=lambda item: item[1])
res_str = ""
for item in sorted_xmin_list:
  res_str += item[0]

print(f"The result is: {res_str}")
