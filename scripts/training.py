
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import click
import os
import torchvision
import torch
import torchvision.transforms as transforms
import functools as ft
import optax
from jax.tree_util import tree_structure, tree_flatten, tree_flatten_with_path
import pprint

from sohl2015.image import Diffusion
from utils import save_model


def neg_loglikelihood(
    mu,
    sigma,
    mu_posterior,
    sigma_posterior,
    trajectory_length,
    covariance_schedule,
    n_colours=1,
):

    # Why the convoluted route? Because the normal route is numerically unstable
    alpha_arr = 1.0 - covariance_schedule
    cumulative_covariance = (1.0 - jnp.exp(jnp.log(alpha_arr).sum()))

    KL = (
        jnp.log(sigma)
        - jnp.log(sigma_posterior)
        + ((sigma_posterior**2 + (mu_posterior - mu) ** 2) / (2 * sigma**2))
        - 0.5
    )

    COMMON = 0.5 * (1 + jnp.log(2 * jnp.pi))

    H_startpoint = COMMON + 0.5 * jnp.log(covariance_schedule[0])
    H_endpoint = COMMON + 0.5 * jnp.log(cumulative_covariance)
    H_prior = COMMON

    negL_bound = KL * trajectory_length + H_startpoint - H_endpoint + H_prior

    negL_gauss = COMMON

    negL_diff = negL_bound - negL_gauss

    L_diff_bits = negL_diff / jnp.log(2.0)

    L_diff_bits_avg = L_diff_bits.mean() * n_colours

    return L_diff_bits_avg


@eqx.filter_value_and_grad
def compute_loss(model_, key, t, images, noise_amp):
    keys = jax.random.split(key, images.shape[0])

    v_forward_diffusion = jax.vmap(model_.forward_diffusion, in_axes=(0,None,0,None))
    v_reverse_diffusion = jax.vmap(model_, in_axes=(None,0))

    image, mu, sigma = v_forward_diffusion(keys, t, images, noise_amp)
    r_mu, r_sigma = v_reverse_diffusion(t, image)
    loss =  neg_loglikelihood(r_mu, r_sigma, mu, sigma, model_.trajectory_length , model_.beta_arr, 1)

    return loss


def named_grad_norms(grads):
    flat = tree_flatten_with_path(grads)[0]
    return {
        ".".join(str(k) for k in path): jnp.sqrt(jnp.sum(leaf**2))
        for path, leaf in flat if leaf is not None
    }

if __name__ == '__main__':

    key = jax.random.PRNGKey(39)

    initial_key, main_key , sampling_key = jax.random.split(key, 3)

    n_hidden_dense_lower_output = 5
    n_hidden_dense_lower = 1000
    n_hidden_dense_upper = 100
    n_hidden_conv = 100
    n_layers_conv = 6
    n_layers_dense_lower = 6
    n_layers_dense_upper = 4

    spatial_width = 28
    n_temporal_basis = 12

    trajectory_length = 500

    model = Diffusion(
        initial_key ,
        n_temporal_basis=n_temporal_basis,
        spatial_width=spatial_width,
        n_hidden_dense_lower_output=n_hidden_dense_lower_output,
        n_hidden_dense_lower=n_hidden_dense_lower,
        n_hidden_dense_upper=n_hidden_dense_upper,
        n_layers_conv=n_layers_conv,
        n_layers_dense_lower=n_layers_dense_lower,
        n_layers_dense_upper=n_layers_dense_upper,
        n_colours=1,
        n_hidden_conv=n_hidden_conv,
        trajectory_length = trajectory_length 
    )

    transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean = 0.1307, std = 0.3081)])

    train_dataset = torchvision.datasets.MNIST("MNIST", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader( train_dataset, batch_size=200, shuffle=True, drop_last=True)

    original_noise_level = 1.0 / 255.  # one pixel intensity step in [0,1] scale
    normalized_noise_level = original_noise_level / 0.5

    learning_rate = 1e-3

    lr_schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=200,    # your batches_per_epoch
            decay_rate=0.97,
            staircase=True
    )

    optimizer = optax.chain( optax.clip_by_global_norm(1.0),      # optional safety like RemoveNotFinite
                             optax.adam(learning_rate=lr_schedule))

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))


    def infinite_trainloader():
        while True:
            yield from trainloader

    steps = 7000

    for step, (batch, y) in zip(range(steps), infinite_trainloader()):

        main_key, sub_key, time_key = jax.random.split(main_key, 3)

        t = jax.random.choice(time_key, jnp.arange(1,  trajectory_length))
        
        loss, grads = compute_loss(model, sub_key,t, batch.numpy(), normalized_noise_level)

        pprint.pp(named_grad_norms(grads))

        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)

        print(f"loss for t={t} is {loss.item()}")

    save_model("mnist_better_hyparameters.ex", model)


