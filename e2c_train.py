import numpy as np
import h5py

import e2c as vae_util

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

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


def kl_normal_loss(qm, q_logv, pm, p_logv):
    # 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    # -0.5 * K.sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=-1)
    kl = -0.5 * (1 - p_logv + q_logv - K.exp(q_logv) / K.exp(p_logv) - K.square(qm - pm) / K.exp(p_logv))
    return K.mean(K.sum(kl, axis=-1))


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

    encoder_ = vae_util.create_encoder(latent_dim, input_shape)
    decoder_ = vae_util.create_decoder(latent_dim, input_shape)
    transition_ = vae_util.create_trans(latent_dim, u_dim)
    sampler_ = vae_util.create_sampler()

    return encoder_, decoder_, transition_, sampler_


# Create plain E2C model and associated loss operations
if __name__ == "__main__":
    
    ################### case specification ######################

#     data_dir = '/data/cees/zjin/lstm_rom/datasets/9W_BHP/'
    data_dir = '/data/cees/zjin/lstm_rom/datasets/9W_BHP_RATE/'
    output_dir = '/data3/Astro/lstm_rom/e2c_larry/saved_models/'
    
#     case_name = '9w_bhp'
    case_name = '9w_bhp_rate'
    
#     suffix = '_single_out_rel_2'
    suffix = '_fix_wl_rel_1'
    
    train_file = case_name + '_e2c_train' + suffix + '_n6600_dt20day_nt22_nrun300.mat'
    eval_file = case_name + '_e2c_eval' + suffix + '_n2200_dt20day_nt22_nrun100.mat'

    #################### model specification ##################
    epoch = 10
    batch_size = 4
    learning_rate = 1e-4
    latent_dim = 50
    u_dim = 9 # control dimension
    
    # load data
    hf_r = h5py.File(data_dir + train_file, 'r')
    sat_t_train = np.array(hf_r.get('sat_t'))
    sat_t1_train = np.array(hf_r.get('sat_t1'))
    bhp_train = np.array(hf_r.get('bhp'))
    hf_r.close()

    num_train = sat_t_train.shape[0]

    hf_r = h5py.File(data_dir + eval_file, 'r')
    sat_t_eval = np.array(hf_r.get('sat_t'))
    sat_t1_eval = np.array(hf_r.get('sat_t1'))
    bhp_eval = np.array(hf_r.get('bhp'))
    hf_r.close()

    # Construct E2C
    input_shape = (60, 60, 1)
    encoder, decoder, transition, sampler = create_e2c(latent_dim, u_dim, input_shape)

    xt = Input(shape=input_shape)
    xt1 = Input(shape=input_shape)
    ut = Input(shape=(u_dim, ))

    zt_mean, zt_logvar = encoder(xt)
    zt = sampler([zt_mean, zt_logvar])
    xt_rec = decoder(zt)

    zt1_mean, zt1_logvar = encoder(xt1)

    # zt1_pred, zt1_mean_pred, zt1_logvar_pred = transition([zt, ut])
    zt1_pred, zt1_mean_pred = transition([zt, zt_mean, ut])
    xt1_pred = decoder(zt1_pred)

    # Compute loss
    loss_rec_t = reconstruction_loss(xt, xt_rec)
    loss_rec_t1 = reconstruction_loss(xt1, xt1_pred)
    loss_kl = kl_normal_loss(zt_mean, zt_logvar, 0., 0.)
    loss_bound = loss_rec_t + loss_rec_t1 + loss_kl

    # loss_trans = kl_normal_loss(zt1_mean_pred, zt1_logvar_pred, zt1_mean, zt1_logvar)

    # Use zt_logvar to approximate zt1_logvar_pred
    loss_trans = kl_normal_loss(zt1_mean_pred, zt_logvar, zt1_mean, zt1_logvar)

    trans_loss_weight = 1.0
    loss = loss_bound + trans_loss_weight * loss_trans

    opt = Adam(lr=learning_rate)

    trainable_weights = encoder.trainable_weights + decoder.trainable_weights + transition.trainable_weights

    updates = opt.get_updates(loss, trainable_weights)

    iterate = K.function([xt, ut, xt1], [loss, loss_rec_t, loss_rec_t1, loss_kl, loss_trans], updates=updates)

    eval_loss = K.function([xt, ut, xt1], [loss])

    num_batch = int(num_train/batch_size)

    for e in range(epoch):
        for ib in range(num_batch):
            ind0 = ib * batch_size
            sat_t_batch = sat_t_train[ind0:ind0+batch_size, ...]
            sat_t1_batch = sat_t1_train[ind0:ind0 + batch_size, ...]
            bhp_batch = bhp_train[ind0:ind0 + batch_size, ...]
            output = iterate([sat_t_batch, bhp_batch, sat_t1_batch])

            # tf.session.run(feed_dict={xt: sat_t_batch, ut: bhp_batch, xt1: sat_t1_batch}, ...
            #                fetches= [loss, loss_rec_t, loss_rec_t1, loss_kl, loss_trans, updates])
            # But output tensor for the updates operation is not returned

            if ib % 10 == 0:
                print('Epoch %d/%d, Batch %d/%d, Loss %f, Loss rec %f, loss rec t1 %f, loss kl %f, loss_trans %f'
                      % (e+1, epoch, ib+1, num_batch, output[0], output[1], output[2], output[3], output[4]))
        eval_loss_val = eval_loss([sat_t_eval, bhp_eval, sat_t1_eval])

        print('Epoch %d/%d, Train loss %f, Eval loss %f' % (e + 1, epoch, output[0], eval_loss_val[0]))

    
    encoder.save_weights(output_dir + 'e2c_encoder_' + case_name + suffix + '_nt%d_l%d_lr%.0e_ep%d.h5' \
                         % (num_train, latent_dim, learning_rate, epoch))
    decoder.save_weights(output_dir + 'e2c_decoder_' + case_name + suffix + '_nt%d_l%d_lr%.0e_ep%d.h5' \
                         % (num_train, latent_dim, learning_rate, epoch))
    transition.save_weights(output_dir + 'e2c_transition_' + case_name + suffix + '_nt%d_l%d_lr%.0e_ep%d.h5' \
                            % (num_train, latent_dim, learning_rate, epoch))














