# Deconvolution/Transpose Convolution - Tensorflow
Function used to reduce the number of lines used to create a transpose convolution in my day to day...

## How to use it?
So you want a deconvolution that can accept batches of [?,im_h,im_w,im_c] using a kernel of 3 by 3 that results in the same shape as the input?

```python
import tensorflow as tf
from deconvolution import deconv

# Load an image of 28 by 28 by 1 ....
im = """LOAD IMAGE"""

# Reshape it to [1,28,28,1]
im = """RESHAPE"""

im_h = 28
im_w = 28
im_c = 1

input_im = tf.placeholder(tf.float32,shape=[None,im_h,im_w,im_c])

ks = 3 # Kernel size
d_shape = [ks,ks,im_c,im_c]
deconv1 = deconv(inp=input_im,
				 shape=d_shape)

with tf.Session as sess:
	feed_dict = {input_im: im}
	out = sess.run(deconv1,feed_dict=feed_dict)
```

Want to upsample by two your initial input of the deconvolution?

```python
import tensorflow as tf
from deconvolution import deconv

# Load an image of 28 by 28 by 1 ....
im = """LOAD IMAGE"""

# Reshape it to [1,28,28,1]
im = """RESHAPE"""

im_h = 28
im_w = 28
im_c = 1

input_im = tf.placeholder(tf.float32,shape=[None,im_h,im_w,im_c])

ks = 3 # Kernel size
d_shape = [ks,ks,im_c,im_c]
deconv1 = deconv(inp=input_im,
				 shape=d_shape,
				 strides=[1,2,2,1])

with tf.Session as sess:
	feed_dict = {input_im: im}
	out = sess.run(deconv1,feed_dict=feed_dict)
```

```
?: Dynamic
im_h: Image height
im_w: Image width
im_c: Image channels
```