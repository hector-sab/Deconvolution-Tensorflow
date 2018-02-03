import tensorflow as tf
def weights(shape,verb=False,name='weights'):
  """
  Description: Weights creation and initialization 

  shape: a list of four elements -> [ker_h,ker_w,im_chan,num_ker]
  verb: Displays the info about the weights tensor
  """
  #w = tf.Variable(tf.truncated_normal(shape, stddev=0.05),name=name)
  # https://arxiv.org/pdf/1502.01852.pdf
  w = tf.Variable(tf.truncated_normal(shape, stddev=np.sqrt(2.0/(
                  shape[0]*shape[1]*shape[2]))),name=name)
  if verb:
    print(w)
  return(w)

def biases(shape,verb=False,name='biases'):
  """
  Description: Biases creation and initialization 

  shape: A list of a single element -> [num_ker]
  verb: Displays the info about the biases tensor
  """
  b = tf.Variable(tf.constant(0.05,shape=shape),name=name)
  if verb:
    print(b)
  return(b)

def deconv2(inp,shape,strides=[1,1,1,1],padding='SAME',relu=False,
  verb=False,name='deconv',dropout=False,drop_prob=0.8,histogram=True,
  l2=False):
  """
  Description: Transpose convolution implementation for CNNs

  inp: Tensor input to be 'deconvoluted'
  shape: List of four elements -> [ker_h,ker_w,out_c,in_c]
  strides: Indicates strides
  padding: 'SAME' - Adds zero padding to the inputs so it generates 
                    an output of the same size
            'VALID' - Does not add zero padding.
  verb: Indicates if the tensor weights/biases info should be displayed
  name: Name of the tensor
  dropout: Indicates if droput is desired
  drop_prob: A number or a scalar tensor indicated the prob of setting
             to zero some weights.
  histogram: Indicates if information for tensorboar is desired to be 
             saved
  l2: l2 loss


  NOTE: If l2 is set to be True, the output will be two tensors. One
        correspondig to the 'deconvolution', and the other used to
        update l2 loss in weights
  """
  with tf.name_scope(name) as scope:
    w = weights(shape,verb=verb)
    b = biases([shape[2]],verb=verb)

    x_shape = tf.shape(inp)
    out_shape = tf.stack([x_shape[0],x_shape[1]*strides[1],
                          x_shape[2]*strides[2],shape[2]])

    transpose_conv = tf.nn.conv2d_transpose(value=inp,
                                            filter=w,
                                            output_shape=out_shape,
                                            strides=strides,
                                            padding=padding)

    transpose_conv += b

    if relu:
      transpose_conv = tf.nn.relu(transpose_conv)
      if histogram:
        tf.summary.histogram('activations',transpose_conv)
    if dropout:
      transpose_conv = tf.nn.dropout(transpose_conv,drop_prob)

    if histogram:
      tf.summary.histogram('weights',w)
      tf.summary.histogram('biases',b)

    if l2:
      l2_reg = tf.nn.l2_loss(w)
      return(transpose_conv,l2_reg)
    else:
      return(transpose_conv)