import numpy as np
import h5py

# import e2c as e2c_util
import e2c_1 as e2c_util


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam


def reconstruction_loss(x, t_decoded):
    '''Reconstruction loss for the plain VAE'''
    v = 0.1
    # return K.mean(K.sum((K.batch_flatten(x) - K.batch_flatten(t_decoded)) ** 2 / (2*v) + 0.5*K.log(2*np.pi*v), axis=-1))
    return K.mean(K.sum((K.batch_flatten(x) - K.batch_flatten(t_decoded)) ** 2 / (2*v), axis=-1))
    # return K.sum((K.batch_flatten(x) - K.batch_flatten(t_decoded)) ** 2, axis=-1)


def l2_reg_loss(qm):
    # 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    # -0.5 * K.sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=-1)
#     kl = -0.5 * (1 - p_logv + q_logv - K.exp(q_logv) / K.exp(p_logv) - K.square(qm - pm) / K.exp(p_logv))
    l2_reg = 0.5*K.square(qm)
    return K.mean(K.sum(l2_reg, axis=-1))


def get_flux_loss(m, state, state_pred):
    # state, state_pred shape (batch_size, 60, 60, 2)
    # p, p_pred shape (batch_size, 60, 60, 1)
    # k shape (batch_size, 60, 60, 1)
    
    # Only consider discrepancies in total flux, not in phases (saturation not used) 
    
    perm = K.exp(m)
    p = K.expand_dims(state[:, :, :, 1], -1)
    p_pred = K.expand_dims(state_pred[:, :, :, 1], -1)

    #print(K.in_shape(xxx))
    
    tran_x = 1./perm[:, 1:, ...] + 1./perm[:, :-1, ...]
    tran_y = 1./perm[:, :, 1:, ...] + 1./perm[:, :, :-1, ...]
    flux_x = (p[:, 1:, ...] - p[:, :-1, ...]) / tran_x
    flux_y = (p[:, :, 1:, :] - p[:, :, :-1, :]) / tran_y
    flux_x_pred = (p_pred[:, 1:, ...] - p_pred[:, :-1, ...]) / tran_x
    flux_y_pred = (p_pred[:, :, 1:, :] - p_pred[:, :, :-1, :]) / tran_y

    loss_x = K.sum(K.abs(K.batch_flatten(flux_x) - K.batch_flatten(flux_x_pred)), axis=-1)
    loss_y = K.sum(K.abs(K.batch_flatten(flux_y) - K.batch_flatten(flux_y_pred)), axis=-1)

    loss_flux = K.mean(loss_x + loss_y)
    return loss_flux


def create_e2c(latent_dim, u_dim, input_shape):
    '''
    Creates a E2C.

    Args:
        latent_dim: dimensionality of latent space
        return_kl_loss_op: whether to return the operation for
                           computing the KL divergence loss.

    Returns:
        The VAE model. If return_kl_loss_op is True, then the
        operation for computing the KL divergence loss is
        additionally returned.
    '''

    encoder_ = e2c_util.create_encoder(latent_dim, input_shape)
    decoder_ = e2c_util.create_decoder(latent_dim, input_shape)
    transition_ = e2c_util.create_trans(latent_dim, u_dim)

    return encoder_, decoder_, transition_


def create_e2c_var_wl(latent_dim, u_dim, input_shape):
    '''
    Creates a E2C.

    Args:
        latent_dim: dimensionality of latent space
        return_kl_loss_op: whether to return the operation for
                           computing the KL divergence loss.

    Returns:
        The VAE model. If return_kl_loss_op is True, then the
        operation for computing the KL divergence loss is
        additionally returned.
    '''

    encoder_, hidden_shapes_ = e2c_util.create_encoder(latent_dim, input_shape)
    decoder_ = e2c_util.create_decoder(latent_dim, input_shape, hidden_shapes_)
    transition_ = e2c_util.create_trans(latent_dim, u_dim)
#     wc_encoder_, = e2c_util.create_wc_encoder(latent_dim, input_shape)
    wc_encoder_, _ = e2c_util.create_encoder(latent_dim, input_shape)
    
    return encoder_, decoder_, transition_, wc_encoder_
