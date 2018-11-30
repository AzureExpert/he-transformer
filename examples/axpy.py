# ==============================================================================
#  Copyright 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

import ngraph_bridge
import numpy as np
import tensorflow as tf

ngraph_bridge.enable()
backend_cpu = 'CPU'
backend_interpreter = 'INTERPRETER'

found_cpu = False
found_interpreter = False
# These will only print when running pytest with flag "-s"
print("Number of supported backends ", ngraph_bridge.backends_len())
supported_backends = ngraph_bridge.list_backends()
print(" ****** Supported Backends ****** ")
for backend_name in supported_backends:
    print(backend_name)
    if backend_name == backend_cpu:
        found_cpu = True
    if backend_name == backend_interpreter:
        found_interpreter = True
print(" ******************************** ")
assert (found_cpu and found_interpreter) == True

print('get_currently_set_backend_name', ngraph_bridge.get_currently_set_backend_name())



a = tf.constant(np.array([[1, 2], [3, 4]]), dtype=np.float32)
b = tf.placeholder(tf.float32, shape=(2, 2))
c = tf.placeholder(tf.float32, shape=())
f = (a + b) * c

with tf.Session() as sess:
    f_val = sess.run(f, feed_dict={b: np.ones((2, 2)), c: np.array(5, )})
    print("Result: ", f_val)

a = tf.constant(2 * np.ones((2, 3)), dtype=np.float32)
b = tf.placeholder(tf.float32, shape=(3, 4))
f = tf.matmul(a, b)

with tf.Session() as sess:
    f_val = sess.run(f, feed_dict={b: 3 * np.ones((3, 4))})
    print("Result: ", f_val)

