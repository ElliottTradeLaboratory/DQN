"""
reference to https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
and https://github.com/lanpa/tensorboard-pytorch
License: Copyleft
"""
import io
import re
import tensorflow as tf
from PIL import Image as Img
import numpy as np

_INVALID_TAG_CHARACTERS = re.compile(r'[^-/\w\.]')

class Logger(object):
    """
        Logging in tensorboard without tensorflow ops.
    """

    def __init__(self, writer):
        self.writer = writer
        self.values = []

    def add_graph(self, graph):
        self.writer.add_graph(graph)

    def add_scalar(self, tag, value):
        tag = clean_tag(tag)

        # Create and write Summary
        self.values += [tf.Summary.Value(tag=tag, simple_value=value)]

    def add_images(self, tag, images):
        """
        Outputs a `Summary` protocol buffer with images.
        The summary has up to `max_images` summary values containing images. The
        images are built from `images` which must be 4-D with shape `[batch_size, height, width,
        channels]` and where `channels` can be:
        *  1: `images` is interpreted as Grayscale.
        *  3: `images` is interpreted as RGB.
        *  4: `images` is interpreted as RGBA.
        The `name` in the outputted Summary.Value protobufs is generated based on the
        name, with a suffix depending on the max_outputs setting:
        *  If `batch_size` is 1, the summary value tag is '*name*/image/0'.
        *  If `batch_size` is greater than 1, the summary value tags are
           generated sequentially as '*name*/image/0', '*name*/image/1', etc.
        Args:
          tag: A name for the generated node. Will also serve as a series name in
            imagesBoard.
          images: A 4-D `uint8` or `float32` `images` of shape `[batch_size, height, width,
            channels]` where `channels` is 1, 3, or 4.
        Returns:
          A scalar `images` of type `string`. The serialized `Summary` protocol
          buffer.
        """
        tag = clean_tag(tag)
        for i, image in enumerate(images):
            image = makenp(image, 'IMG')
            image = image.astype(np.float32)
            image = (image*255).astype(np.uint8)
            image = make_image(image)
            self.values += [tf.Summary.Value(tag='{}/image/{}'.format(tag, i), image=image)]


    def add_histogram(self, tag, values, bins=1000):
        """Logs the histogram of a list/vector of values."""
        tag = clean_tag(tag)

        # Convert to a numpy array
        values = np.array(values)
        
        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/imagesflow/imagesflow/blob/master/imagesflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        self.values += [tf.Summary.Value(tag=tag, histo=hist)]


    def flush(self, step):
        if len(self.values) > 0:
            summaries = tf.Summary(value=self.values)
            self.writer.add_summary(summaries, step)
            self.writer.flush()
            self.values =[]

def clean_tag(name):
  # In the past, the first argument to summary ops was a tag, which allowed
  # arbitrary characters. Now we are changing the first argument to be the node
  # name. This has a number of advantages (users of summary ops now can
  # take advantage of the tf name scope system) but risks breaking existing
  # usage, because a much smaller set of characters are allowed in node names.
  # This function replaces all illegal characters with _s, and logs a warning.
  # It also strips leading slashes from the name.
  if name is not None:
    new_name = _INVALID_TAG_CHARACTERS.sub('_', name)
    new_name = new_name.lstrip('/')  # Remove leading slashes
    if new_name != name:
      logging.info(
          'Summary name %s is illegal; using %s instead.' %
          (name, new_name))
      name = new_name
  return name

modes = {1:'L', 3:'RGB', 4:'RGBA'}
def make_image(tensor):
    """Convert an numpy representation image to Image protobuf"""
    from PIL import Image
    if len(tensor.shape) == 2:
        height, width = tensor.shape
        channel = 1
    else:
        height, width, channel = tensor.shape
        if channel == 1:
            tensor = tensor.reshape(height, width)
    assert channel in modes, "channel '{}' is not supported. image shape={}".format(channel, tensor.shape)
    mode = modes[channel]
    image = Image.fromarray(tensor, mode=mode)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)

def makenp(x, modality=None):
    # if already numpy, return
    if isinstance(x, np.ndarray):
        if modality == 'IMG' and x.dtype == np.uint8:
            return x.astype(np.float32)/255.0
        return x
    if np.isscalar(x):
        return np.array([x])
    if 'torch' in str(type(x)):
        return pytorch_np(x, modality)
    if 'chainer' in str(type(x)):
        return chainer_np(x, modality)
    if 'mxnet' in str(type(x)):
        return mxnet_np(x, modality)

def pytorch_np(x, modality):
    import torch
    if isinstance(x, torch.autograd.variable.Variable):
        x = x.data
    x = x.cpu().numpy()
    if modality == 'IMG':
        x = _prepare_image(x)
    return x


def theano_np(x):
    import theano
    pass

def caffe2_np(x):
    pass

def mxnet_np(x, modality):
    x = x.asnumpy()
    if modality == 'IMG':
        x = _prepare_image(x)
    return x    

def chainer_np(x, modality):
    import chainer
    x = chainer.cuda.to_cpu(x.data)
    if modality == 'IMG':
        x = _prepare_image(x)
    return x

def make_grid(I, ncols=8):
    assert isinstance(I, np.ndarray), 'plugin error, should pass numpy array here'
    assert I.ndim==4 and I.shape[1]==3
    nimg = I.shape[0]
    H = I.shape[2]
    W = I.shape[3]
    ncols = min(nimg, ncols)
    nrows = int(np.ceil(float(nimg)/ncols))
    canvas = np.zeros((3, H*nrows, W*ncols))
    i = 0
    for y in range(nrows):
        for x in range(ncols):
            if i>=nimg:
                break
            canvas[:, y*H:(y+1)*H, x*W:(x+1)*W] = I[i]
            i = i+1
    return canvas
def _prepare_image(I):
    assert isinstance(I, np.ndarray), 'plugin error, should pass numpy array here'
    assert I.ndim==2 or I.ndim==3 or I.ndim==4
    if I.ndim==4: #NCHW
        if I.shape[1]==1: #N1HW
            I = np.concatenate((I,I,I), 1) #N3HW
        assert I.shape[1]==3            
        I = make_grid(I) #3xHxW
    if I.ndim==3 and I.shape[0]==1: #1xHxW
        I = np.concatenate((I,I,I), 0) #3xHxW
    if I.ndim==2: #HxW
        I = np.expand_dims(I, 0) #1xHxW
        I = np.concatenate((I,I,I), 0) #3xHxW
    I = I.transpose(1,2,0)
    
    return I