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

import ngraph
import numpy as np
import tensorflow as tf
import time


def gemm_trial(n, n_zeros=0, n_ones=0, fp=None):
    a = tf.constant(np.float32(np.random.randint(low=-5, high=5, size=(n,n))), dtype=np.float32)
    b = tf.placeholder(tf.float32, shape=(n, n))
    rb = np.float32(np.random.randint(low=-5, high=5, size=(n,n)))
    c = tf.placeholder(tf.float32, shape=())
    f = tf.matmul(a, b) * c

    with tf.Session() as sess:
        t0 = time.time()
        f_val = sess.run(f, feed_dict={b: rb, c: 3.})
        t1 = time.time()-t0
        print("Result: ", f_val)
        print("Time in sec: ", t1)

if __name__ == "__main__":
    gemm_trial(20)
    
