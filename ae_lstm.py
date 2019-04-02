from layers import *

from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, BatchNormalization, LSTM
from keras.models import Model
from keras.layers.merge import Add, Multiply, Concatenate

def create_trans(latent_dim, u_dim):
    '''
    Creates a linear transition model in latent space.

    '''
    zt = Input(shape=(latent_dim,))
    dt = Input(shape=(1,))
    ut = Input(shape=(u_dim,))
    
    zt_expand = Concatenate(axis=-1)([zt, dt])
    
    

    trans_encoder = create_trans_encoder(latent_dim + 1)
    hz = trans_encoder(zt_expand)

    At = Dense(latent_dim*latent_dim)(hz)
    At = Reshape((latent_dim, latent_dim))(At)

    Bt = Dense(latent_dim*u_dim)(hz)
    Bt = Reshape((latent_dim, u_dim))(Bt)

    batch_dot_layer = Lambda(lambda x: K.batch_dot(x[0], x[1]))
    
    scalar_multi = Lambda(lambda x: x[0] * x[1]) # Larry Jin
    ut_dt = scalar_multi([ut, dt]) # Larry Jin
    
    
#     zt1 = Add()([batch_dot_layer([At, zt]), batch_dot_layer([Bt, ut])]) 
    zt1 = Add()([batch_dot_layer([At, zt]), batch_dot_layer([Bt, ut_dt])]) # Larry Jin
    
    # At*zt + Bt*ut
    
    # what if I want: At*zt + Bt*ut*dt

    trans = Model([zt, ut, dt], [zt1])

    return trans

def create_lstm(latent_dim, u_dim):
    '''
    Creates a LSTM transition model in latent space.

    '''
    zt = Input(shape=(latent_dim,))
    dt = Input(shape=(1,))
    ut = Input(shape=(u_dim,))
    
    zt_expand = Concatenate(axis=-1)([zt, ut, dt])
    
    
    l1 = lstm(250)(zt_expand)
    l2 = lstm(250)(l1)
    zt1 = Dense(latent_dim)(l2)
    
    # At*zt + Bt*ut
    
    # what if I want: At*zt + Bt*ut*dt

    lstm_trans = Model([zt, ut, dt], [zt1])

    return lstm_trans

def create_trans_encoder(input_dim):
    '''
    Creates FC transition model.

    '''
    
    zt = Input(shape=(input_dim,))

    # Embed z to hz
    hidden_dim = 200
    hz = fc_bn_relu(hidden_dim)(zt)
    hz = fc_bn_relu(hidden_dim)(hz)
    hz = fc_bn_relu(input_dim-1)(hz) # make sure dim of hz is consistent with 

    trans_encoder = Model(zt, hz)

    return trans_encoder


def create_encoder(latent_dim, input_shape):
    '''
    Creates a convolutional encoder model.

    '''
    encoder_iput = Input(shape=input_shape, name='image')

    x = conv_bn_relu(16, 3, 3, stride=(2, 2))(encoder_iput)
    x = conv_bn_relu(32, 3, 3, stride=(1, 1))(x)
    x = conv_bn_relu(64, 3, 3, stride=(2, 2))(x)
    x = conv_bn_relu(128, 3, 3, stride=(1, 1))(x)

    for i in range(3):
        x = res_conv(128, 3, 3)(x)

    x = Flatten()(x)

    xi = Dense(latent_dim, name='t_mean')(x)
#     t_log_var = Dense(latent_dim, name='t_log_var')(x)

    return Model(encoder_iput, xi, name='encoder')


def create_decoder(latent_dim, input_shape):
    '''
    Creates a (trans-)convolutional decoder model.


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
    y = Conv2D(input_shape[2], (3, 3), padding='same', activation=None)(x)

    return Model(decoder_input, y, name='decoder')


# def sample(args):
#     '''
#     Draws samples from a standard normal and scales the samples with
#     standard deviation of the variational distribution and shifts them
#     by the mean.

#     Args:
#         args: sufficient statistics of the variational distribution.

#     Returns:
#         Samples from the variational distribution.
#     '''
#     t_mean, t_log_var = args
#     t_sigma = K.sqrt(K.exp(t_log_var))
#     epsilon = K.random_normal(shape=K.shape(t_mean), mean=0., stddev=1.)
# #     return t_mean + t_sigma * epsilon
#     return t_mean + t_sigma * 0


# def create_sampler():
#     '''
#     Creates a sampling layer.
#     '''
#     return Lambda(sample, name='sampler')


# ---------------------------------------------------
#
# modified by Larry from Yimin's first implementation
#
# ---------------------------------------------------

# def sample_normal(mu, log_var):
#     sigma = K.sqrt(K.exp(log_var))
#     epsilon = K.random.normal(shape=K.shape(mu), mean=0., stddev=1.)
#     return mu + sigma * epsilon

