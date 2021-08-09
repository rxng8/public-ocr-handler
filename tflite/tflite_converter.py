# import tensorflow as tf

# from argparse import ArgumentParser

# if __name__ == '__main__':
  
#   parser = ArgumentParser(description="TF lite converter")

#   parser.add_argument("-s", "--source", type=str, help="Path to source saved model.", required=True)
#   parser.add_argument("-t", "--target", type=str, help="Path to target tflite model", required=True)
#   args = parser.parse_args()

#   # Convert the model
#   converter = tf.lite.TFLiteConverter.from_saved_model(args.source) # path to the SavedModel directory
#   converter.allow_custom_ops=True
#   tflite_model = converter.convert()

#   # Save the model.
#   with open(args.target, 'wb') as f:
#     f.write(tflite_model)

# %%

import tensorflow as tf

# SOURCE = '.\exported-models\ssd_resnet50_v1_fpn\saved_model'
SOURCE = '.\exported-models\ssd_mobilenet_v2_fpnlite\saved_model'
TARGET = '.\exported-models/detect.tflite'

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(SOURCE) # path to the SavedModel directory
converter.allow_custom_ops=True

# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
#                                       tf.lite.OpsSet.TFLITE_BUILTINS]

'''
Quantization: More information regards:
  https://www.tensorflow.org/lite/performance/model_optimization
and
  https://www.tensorflow.org/lite/performance/post_training_quantization
'''
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# def representative_dataset_gen():
#     for _ in range(num_calibration_steps):
#   # Get sample input data as a numpy array in a method of your choosing.
#   yield [input]
# converter.representative_dataset = representative_dataset_gen

tflite_model = converter.convert()

"""
Knowledge about custom ops when tf supports but tflite is not supported:
  https://www.tensorflow.org/lite/convert#other_features
then
  https://www.tensorflow.org/lite/guide/ops_custom
"""

# Save the model.
with open(TARGET, 'wb') as f:
  f.write(tflite_model)

