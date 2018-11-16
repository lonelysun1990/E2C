from layers import *

from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape
from keras.models import Model


def create_encoder(latent_dim, input_shape):
    '''
    Creates a convolutional encoder model for MNIST images.

    - Input for the created model are MNIST images.
    - Output of the created model are the sufficient statistics
      of the variational distriution q(t|x;phi), mean and log
      variance.
    '''
    encoder_iput = Input(shape=input_shape, name='image')

    x = conv_bn_relu(16, 3, 3, stride=(2, 2))(encoder_iput)
    x = conv_bn_relu(32, 3, 3, stride=(1, 1))(x)
    x = conv_bn_relu(64, 3, 3, stride=(2, 2))(x)
    x = conv_bn_relu(128, 3, 3, stride=(1, 1))(x)

    for i in range(3):
        x = res_conv(128, 3, 3)(x)

    x = Flatten()(x)

    t_mean = Dense(latent_dim, name='t_mean')(x)
    t_log_var = Dense(latent_dim, name='t_log_var')(x)

    return Model(encoder_iput, [t_mean, t_log_var], name='encoder')


def create_decoder(latent_dim, input_shape):
    '''
    Creates a (de-)convolutional decoder model for MNIST images.

    - Input for the created model are latent vectors t.
    - Output of the model are images of shape (28, 28, 1) where
      the value of each pixel is the probability of being white.
    '''
    decoder_input = Input(shape=(latent_dim,), name='t')

    x = Dense(int(input_shape[0] * input_shape[1]/16*128), activation='relu')(decoder_input)

    x = Reshape((int(input_shape[0]/4), int(input_shape[1]/4), 128))(x)

    for i in range(3):
        x = res_conv(128, 3, 3)(x)

    x = dconv_bn_nolinear(128, 3, 3, stride=(1, 1))(x)
    x = dconv_bn_nolinear(64, 3, 3, stride=(2, 2))(x)
    x = dconv_bn_nolinear(32, 3, 3, stride=(1, 1))(x)
    x = dconv_bn_nolinear(16, 3, 3, stride=(2, 2))(x)
    y = Conv2D(1, (3, 3), padding='same', activation=None)(x)

    return Model(decoder_input, y, name='decoder')


def sample(args):
    '''
    Draws samples from a standard normal and scales the samples with
    standard deviation of the variational distribution and shifts them
    by the mean.

    Args:
        args: sufficient statistics of the variational distribution.

    Returns:
        Samples from the variational distribution.
    '''
    t_mean, t_log_var = args
    t_sigma = K.sqrt(K.exp(t_log_var))
    # epsilon = K.random_normal(shape=K.shape(t_mean), mean=0., stddev=1.)
    return t_mean + t_sigma * 0


def create_sampler():
    '''
    Creates a sampling layer.
    '''
    return Lambda(sample, name='sampler')

