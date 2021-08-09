# %%

import numpy as np
import os
from tensorflow.python.ops.gen_math_ops import Imag

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from PIL import Image

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

CSV_PATH = "./images/_annotations_new.csv"
IMAGE_DIR = "./images/"

spec = model_spec.get('efficientdet_lite0')


# %%

train_data, validation_data, test_data = object_detector.DataLoader.from_csv(
  CSV_PATH,
  IMAGE_DIR,
)

# %%

model = object_detector.create(
    train_data, 
    model_spec=spec, 
    batch_size=8, 
    train_whole_model=True,
    validation_data=validation_data,
    epochs=100,
)

# %%

model.evaluate(test_data)


# %%

model.export(export_dir='.')

# %%

model.evaluate_tflite('model_det0_1.tflite', test_data)

# %%


import cv2

from PIL import Image

model_path = 'model_det0_1.tflite'

# Load the labels into a list
# classes = ['???'] * model.model_spec.config.num_classes
classes = ['number']
# label_map = model.model_spec.config.label_map
# for label_id, label_name in label_map.as_dict().items():
#   classes[label_id-1] = label_name

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
    label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
    cv2.putText(original_image_np, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_image.numpy().astype(np.uint8), results, original_uint8

# %%

INPUT = "../../test.jpg"
INPUT_2 = "images/3_jpg.rf.de17a1ccf5a2427d4b9fd6208e6bf4cb.jpg"
INPUT_3 = "images/13_png.rf.8f6313e47786e4af78b0f4e3de5afa50.jpg"

DETECTION_THRESHOLD = 0.3
# NEW_INPUT = "../../test_1.jpg"

# im = Image.open(INPUT)
# im.thumbnail((512, 512), Image.ANTIALIAS)
# im.save(NEW_INPUT, 'PNG')

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
print(input_height)
print(input_width)

# Run inference and draw detection result on the local copy of the original file
original_img, results, detection_result_image = run_odt_and_draw_results(
  INPUT_2,
  interpreter,
  threshold=DETECTION_THRESHOLD
)

# Show the detection result
Image.fromarray(detection_result_image)

# %%

# Cropped

for fname in os.listdir("./images"):
  
  if ".png" in fname[-4:] or ".jpg" in fname[-4:]:
    print(fname)
    # Run inference and draw detection result on the local copy of the original file
    original_img, results, detection_result_image = run_odt_and_draw_results(
      os.path.join("images/", fname),
      interpreter,
      threshold=DETECTION_THRESHOLD
    )

    for i, obj in enumerate(results):
      # Convert the object bounding box from relative coordinates to absolute
      # coordinates based on the original image resolution
      ymin, xmin, ymax, xmax = obj['bounding_box']
      xmin = int(xmin * original_img.shape[1])
      xmax = int(xmax * original_img.shape[1])
      ymin = int(ymin * original_img.shape[0])
      ymax = int(ymax * original_img.shape[0])

      cropped_img = Image.fromarray(original_img[ymin:ymax, xmin:xmax])
      cropped_img.save(os.path.join("cropped_images/", fname[:-4] + "_" + str(i) + fname[-4:]))

    print(f"Done {fname}\n")

# %%

import matplotlib.pyplot as plt

# masked
cnt = 0
for fname in os.listdir("./images"):
  
  if ".png" in fname[-4:] or ".jpg" in fname[-4:]:
    print(fname)
    # Run inference and draw detection result on the local copy of the original file
    original_img, results, detection_result_image = run_odt_and_draw_results(
      os.path.join("images/", fname),
      interpreter,
      threshold=DETECTION_THRESHOLD
    )

    for i, obj in enumerate(results):
      # Convert the object bounding box from relative coordinates to absolute
      # coordinates based on the original image resolution
      ymin, xmin, ymax, xmax = obj['bounding_box']
      xmin = max(0, int(xmin * original_img.shape[1]))
      xmax = min(original_img.shape[1], int(xmax * original_img.shape[1]))
      ymin = max(0, int(ymin * original_img.shape[0]))
      ymax = min(original_img.shape[0], int(ymax * original_img.shape[0]))

      print(f"width: {original_img.shape[1]}, height: {original_img.shape[0]}")
      print(f"xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}")

      cropped_img = original_img[ymin:ymax, xmin:xmax]
      # plt.imshow(cropped_img)
      # plt.show()

      # We want to pad the result to ratio 1:1. So
      # the scale to input size 320 x 320 does not distort the
      # input!
      padded_img = tf.image.resize_with_pad(
        cropped_img, 
        xmax - xmin, 
        xmax - xmin, 
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      ).numpy()
      # plt.imshow(padded_img)
      # plt.show()
      padded_img = Image.fromarray(padded_img)
      padded_img.save(os.path.join("masked_images/", str(cnt) + "_" + str(i) + fname[-4:]))
    
    print(f"Done {fname}\n")
    cnt += 1

# %%
import cv2
import matplotlib.pyplot as plt

# Test integrity of unpadded images
UNPADDED_PATH = "./cropped_images/6217710702599812.jpg"
img = cv2.imread(UNPADDED_PATH)
img = np.asarray(img)

resized = tf.image.resize(
  img,
  (320, 320),
  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
)

plt.imshow(resized)
plt.show()


# %%

# Now test with the padded images
PADDED_PATH = "./masked_images/50_0.jpg"

img = cv2.imread(PADDED_PATH)
img = np.asarray(img)

resized = tf.image.resize(
  img,
  (320, 320),
  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
)

plt.imshow(resized)
plt.show()
