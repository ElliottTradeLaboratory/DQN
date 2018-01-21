import numpy as np


class Scale(object):
    def __init__(self, height, width, args):
        self.height = height
        self.width = width
        self.to_shape = (width, height)
        self.args = args
        self.gain = args.get('gain', 1)
        self.gain_level = args.get('gain_level', 1)
        self.normalized = args.get('screen_normalize', 'env') == 'env'

    def forward(self, x):
        if x.ndim > 3:
            x = x[0]

        return self._forward(x)

    def _forward(self, x):
        raise NotImplementedError()

    def _assertKeyError(self):
        assert False, "inter:'{}' is not support in {}".format(self.args.preproc, self.args.inter)

    def show(self, x):
        import cv2
        cv2.imshow("show", cv2.cvtColor(x, cv2.COLOR_BGR2RGB) if len(x.shape) == 3 else x)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def rgb2y(self, rgb):
        y_image = 0.299 * rgb[..., 0]
        y_image[:] += 0.587 * rgb[..., 1]
        y_image[:] += 0.114 * rgb[..., 2]
        return y_image

class ScaleImage(Scale):
    def __init__(self, height, width, args):
        super(ScaleImage, self).__init__(height, width, args)
        import lutorpy as lua
        lua.LuaRuntime(zero_based_index=True)
        self.torch = require('torch')
        self.image = require('image')
        print('image processor is use torch image')

    def _forward(self, x):

        x = np.transpose(x, (2,0,1))
        x = self.torch.fromNumpyArray(x)
        x = self.image.rgb2y(x)
        x = self.image.scale(x, self.width, self.height, 'bilinear')
        x = x.asNumpyArray()

        #self.show(x.reshape(84,84,1))

        return x.squeeze()

class ScaleCV2(Scale):
    def __init__(self, height, width, args):
        super(ScaleCV2, self).__init__(height, width, args)
        assert args.inter is not None
        import cv2
        try:
            self.inter = {'NEAREST' :cv2.INTER_NEAREST,
                          'LINEAR'  :cv2.INTER_LINEAR,
                          'AREA'    :cv2.INTER_AREA,
                          'CUBIC'   :cv2.INTER_CUBIC,
                          'LANCZOS4':cv2.INTER_LANCZOS4}.get(args.inter)
        except KeyError:
            self._assertKeyError()

        assert self.inter is not None
        self.cv2 = cv2

    def _forward(self, x):
        assert self.inter is not None
        x = self.cv2.cvtColor(x, self.cv2.COLOR_RGB2GRAY)
        x = self.cv2.resize(x, self.to_shape, self.inter)
        return x

class ScaleTensorflow(Scale):
    def __init__(self, height, width, args):
        super(ScaleTensorflow, self).__init__(height, width, args)
        assert args.inter is not None

        from tensorflow_extensions import get_session
        import tensorflow as tf

        with tf.device('/cpu:0'):
            with tf.name_scope('ScaleTensorflow'):
                resize_funcs = {'NEAREST' :tf.image.resize_nearest_neighbor,
                                'LINEAR'  :tf.image.resize_bilinear,
                                'AREA'    :tf.image.resize_area,
                                'CUBIC'   :tf.image.resize_bicubic}
                self.img = tf.placeholder(np.float32, [1, 210, 160, 3])
                if self.normalized:
                    img = self.img * 255.0
                else:
                    img = self.img
                x = tf.image.rgb_to_grayscale(img)
                try:
                    resized_img = resize_funcs[args.inter](x, [height, width])
                except KeyError:
                    self._assertKeyError()

                if self.normalized:
                    self.resize_op = resized_img / 255.0
                else:
                    self.resize_op = resized_img
        self.sess = get_session()

    def _forward(self, x):
        x = self.sess.run(self.resize_op, feed_dict={self.img: [x]})
        x = x[0].reshape((self.height, self.width))

        return x

    def __del__(self):
        self.sess = None

class ScalePIL(Scale):
    def __init__(self, height, width, args):
        super(ScalePIL, self).__init__(height, width, args)
        assert args.inter is not None
        import PIL.Image as image
        self.image = image
        try:
            self.inter = {'NEAREST' :image.NEAREST,
                          'LINEAR'  :image.BILINEAR,
                          'BOX'     :image.BOX,
                          'CUBIC'   :image.BICUBIC,
                          'LANCZOS4':image.LANCZOS,
                          'HAMMING' :image.HAMMING}.get(args.inter)
        except KeyError:
            self._assertKeyError()

        assert self.inter is not None

    def _forward(self, x):
        x = np.uint8(x * 255.0)  if self.normalized else \
            np.uint8(x)
        x = self.image.fromarray(x)
        x = x.convert("L")
        x = x.resize(self.to_shape, self.inter)
        x = np.array(x).astype(np.float32)

        return x / 255.0 if self.normalized else x


class ScaleScikit(Scale):
    def __init__(self, height, width, args):
        super(ScaleScikit, self).__init__(height, width, args)
        assert args.inter is not None
        from skimage.transform import resize
        self.resize = resize
        try:
            self.inter = {'NEAREST'  :0,
                          'LINEAR'   :1,
                          'QUADRATIC':2,
                          'CUBIC'    :3,
                          'QUARTIC'  :4,
                          'QUINTIC'  :5}.get(args.inter)
        except KeyError:
            self._assertKeyError()

        assert self.inter is not None

    def _forward(self, x):
        # scikit-image does not have grayscale function
        # as Y = 0.299R + 0.587G + 0.114B.
        x = self.rgb2y(x)
        x = self.resize(x, (self.height, self.width), order=self.inter)
        return x

class ScaleMix(Scale):
    def __init__(self, height, width, args):
        super(ScaleMix, self).__init__(height, width, args)
        preproc1, preproc2 = args.preproc.split(':')
        self.scale1 = get_preprocess(preproc1)(height, width, args)
        self.scale2 = get_preprocess(preproc2)(height, width, args)
        print('use ScaleMIX gain:{} gain_level:{}'.format(self.gain, self.gain_level))

    def _forward(self, x):
        x1 = self.scale1.forward(x)
        x2 = self.scale2.forward(x)
        x = np.maximum(x1, x2)
        if self.gain > 1:
            x[x < self.gain_level] *= self.gain
        return x

def get_preprocess(type):

    _preproceses = {'cv2'       : ScaleCV2,
                    'image'     : ScaleImage,
                    'PIL'       : ScalePIL,
                    'scikit'    : ScaleScikit,
                    'tensorflow': ScaleTensorflow}
    if not type in _preproceses:
        preprocs = type.split(':')
        assert len(preprocs) == 2, '{} is not support.'.format(type)
        for preproc in preprocs:
            assert preproc in _preproceses, '{} is not support.'.format(preproc)
        return ScaleMix

    return _preproceses.get(type)
