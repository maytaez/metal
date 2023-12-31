# %%
import tensorflow as tf

tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
# %%
with tf.Session() as sess:
    devices = sess.list_devices()
# %%
import tensorflow as tf

assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()
# %%
import tensorflow as tf

if tf.test.gpu_device_name():
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
# %%
import sys

from tensorflow import keras
import pandas as pd
import sklearn as sk
import scipy as sp
import tensorflow as tf
import platform

print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
# print(f"Keras Version: {keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print(f"SciPy {sp.__version__}")
gpu = len(tf.config.list_physical_devices("GPU")) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
# %%
