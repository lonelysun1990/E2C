import numpy as np
import h5py

import vae as vae_util

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras import backend as K
from keras import layers
from keras.models import Model
from keras.optimizers import Adam


def vae_loss(x, t_decoded):
    '''Total loss for the plain VAE'''
    return K.mean(reconstruction_loss(x, t_decoded) + vae_kl_loss)
    # return K.mean(reconstruction_loss(x, t_decoded))


def reconstruction_loss(x, t_decoded):
    '''Reconstruction loss for the plain VAE'''
    v = 0.1
    return K.sum((K.batch_flatten(x) - K.batch_flatten(t_decoded)) ** 2 / (2*v) + 0.5*K.log(2*np.pi*v), axis=-1)
    # return K.sum((K.batch_flatten(x) - K.batch_flatten(t_decoded)) ** 2, axis=-1)


def create_vae(latent_dim):
    '''
    Creates a VAE able to auto-encode MNIST images.

    Args:
        latent_dim: dimensionality of latent space
        return_kl_loss_op: whether to return the operation for
                           computing the KL divergence loss.

    Returns:
        The VAE model. If return_kl_loss_op is True, then the
        operation for computing the KL divergence loss is
        additionally returned.
    '''

    input_shape = (60, 60, 1)

    encoder = vae_util.create_encoder(latent_dim, input_shape)
    decoder = vae_util.create_decoder(latent_dim, input_shape)
    sampler = vae_util.create_sampler()

    x = layers.Input(shape=input_shape)
    t_mean, t_log_var = encoder(x)
    t = sampler([t_mean, t_log_var])
    t_decoded = decoder(t)

    model = Model(x, t_decoded, name='vae')

    # kl_loss = -0.5 * K.sum(1 + t_log_var\
    #                        - K.square(t_mean)\
    #                        - K.exp(t_log_var), axis=-1)

    kl_loss = 0.5 * K.sum(K.square(t_mean), axis=-1)

    return model, encoder, decoder, kl_loss


# Create plain VAE model and associated KL divergence loss operation
if __name__ == "__main__":
    latent_dim = 50

    data_dir = '/data3/Astro/lstm_rom/explore/data/9w_bhp_wl/'

    hf_r = h5py.File(data_dir + 'pres_sat.mat', 'r')
    sat = np.array(hf_r.get('sat'))
    hf_r.close()

    # sat (num_run * num_step, nx, ny)
    nr = 10000
    sat = sat.reshape((nr, 60, 60, 1))
    shuffle_ind = np.random.permutation(nr)
    sat = sat[shuffle_ind,:, :, :]

    num_train = 3000
    num_eval = 10
    sat_train = sat[:num_train, ...]
    sat_eval = sat[-num_eval:, ...]

    # Construct VAE
    vae_model, encoder, decoder, vae_kl_loss = create_vae(latent_dim)
    opt = Adam(lr=1e-3)
    loss = vae_loss(vae_model.input, vae_model.output)
    vae_model.compile(optimizer=opt, loss='mse')
    vae_model.summary()

    updates = opt.get_updates(loss, vae_model.trainable_weights)

    iterate = K.function(vae_model.inputs, [loss], updates=updates)

    eval_loss = K.function(vae_model.inputs, [loss])

    # loss = vae_loss(sat_eval, sat_rec_eval)
    epoch = 10

    batch_size = 4

    num_batch = int(num_train/batch_size)

    for e in range(epoch):
        for ib in range(num_batch):
            ind0 = ib * batch_size
            sat_batch = sat_train[ind0:ind0+batch_size, ...]
            loss_val = iterate([sat_batch])
            if ib % 10 == 0:
                print('Epoch %d/%d, Batch %d/%d, Loss %f' % (e+1, epoch, ib+1, num_batch, loss_val[0]))
        eval_loss_val = eval_loss([sat_eval])
        print('Epoch %d/%d, Train loss %f, Eval loss %f' % (e + 1, epoch, loss_val[0], eval_loss_val[0]))

    # vae_model.fit(x=sat_train, y=sat_train, epochs=epoch, shuffle=True, validation_data=(sat_eval, sat_eval), verbose=2, batch_size=4)

    output_dir = '/data3/Astro/lstm_rom/explore/saved_models/'
    vae_model.save_weights(output_dir + 'ae_deep_bn_gaussian_mse_nt%d_l%d_lr1e-3_ep%d.h5' % (num_train, latent_dim, epoch))














