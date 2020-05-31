from lunzi import nn
import tensorflow as tf
import numpy as np

__all__ = ['NatureCNN', 'FCLayer', 'ortho_initializer']

# def nature_cnn(unscaled_images):
#     """
#     CNN from Nature paper.
#     """
#     scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
#     activ = tf.nn.relu
#     h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
#     h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
#     h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
#     h3 = conv_to_fc(h3)
#     return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


def ortho_initializer(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


class ConvLayer(nn.Module):
    def __init__(self, nin, nf, rf, stride, padding='VALID', init_scale=1.0):
        super().__init__()
        self.strides = [1, stride, stride, 1]
        self.padding = padding

        w_shape = [rf, rf, nin, nf]
        b_shape = [1, 1, 1, nf]
        self.w = nn.Parameter(ortho_initializer(init_scale)(w_shape, np.float32), dtype=tf.float32, name="w")
        self.b = nn.Parameter(tf.constant_initializer(0.0)(b_shape), dtype=tf.float32, name="b")

    def forward(self, x):
        return self.b + tf.nn.conv2d(x, self.w, strides=self.strides, padding=self.padding)


class FCLayer(nn.Module):
    def __init__(self, nin, nh, init_scale=1., init_bias=0.):
        super().__init__()
        self.w = nn.Parameter(ortho_initializer(init_scale)([nin, nh], np.float32), "w")
        self.b = nn.Parameter(tf.constant_initializer(init_bias)([nh]), "b")

    def forward(self, x):
        return tf.matmul(x, self.w) + self.b


class BaseCNN(nn.Module):
    def __init__(self, nin, hidden_sizes=(32, 64, 64,), kernel_sizes=(8, 4, 3), strides=(4, 2, 1), init_scale=np.sqrt(2)):
        super().__init__()

        assert len(hidden_sizes) == len(kernel_sizes) == len(strides)
        layer = []
        for i in range(len(hidden_sizes)):
            nf, rf, stride = hidden_sizes[i], kernel_sizes[i], strides[i]
            layer.append(ConvLayer(nin, nf, rf, stride, init_scale=init_scale))
            layer.append(nn.ReLU())
            nin = nf
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        x = self.layer(x)
        return x


class NatureCNN(nn.Module):
    def __init__(self, n_channel: int):
        super().__init__()
        self.net = BaseCNN(n_channel)
        self.initialized = False

    def forward(self, x):
        x = self.net(x)
        x = tf.layers.flatten(x)
        if not self.initialized:
            layer = [
                FCLayer(nin=x.shape[-1].value, nh=512, init_scale=np.sqrt(2)),
                nn.ReLU()
                ]
            self.conv_to_fc = nn.Sequential(*layer)
            self.initialized = True
        x = self.conv_to_fc(x)
        return x





