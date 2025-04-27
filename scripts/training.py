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
import yaml

import sohl2015.image as image
import sohl2015.spiral as spiral_
from utils import save_model, infinite_trainloader



def named_grad_norms(grads):
    flat = tree_flatten_with_path(grads)[0]
    return {
        ".".join(str(k) for k in path): jnp.sqrt(jnp.sum(leaf**2))
        for path, leaf in flat if leaf is not None
    }
    

@click.group()
def cli():
    """CLI for training or running inference."""
    pass


@cli.command()
@click.option("--seed", default=1900, help="seed.")
@click.option(
    "--config",
    default="./configs/training/mnist.yaml",
    help="Filename to save model checkpoint",
)
def sohl(seed, config):
    key = jax.random.PRNGKey(seed)

    initial_key, main_key , sampling_key = jax.random.split(key, 3)

    with open(config, 'r') as file:
        config = yaml.safe_load(file)
    
    model = image.Diffusion(
        initial_key ,
        n_temporal_basis=config['n_temporal_basis'],
        spatial_width=config['spatial_width'],
        n_hidden_dense_lower_output=config['n_hidden_dense_lower_output'],
        n_hidden_dense_lower=config['n_hidden_dense_lower'],
        n_hidden_dense_upper=config['n_hidden_dense_upper'],
        n_layers_conv=config['n_layers_conv'],
        n_layers_dense_lower=config['n_layers_dense_lower'],
        n_layers_dense_upper=config['n_layers_dense_upper'],
        n_colours=config['n_colours'],
        n_hidden_conv=config['n_hidden_conv'],
        trajectory_length = config['trajectory_length'] 
    )

    transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean = 0.1307, std = 0.3081)])

    train_dataset = torchvision.datasets.MNIST("MNIST", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader( train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    original_noise_level = 1.0 / 255.  # one pixel intensity step in [0,1] scale
    normalized_noise_level = original_noise_level / 0.5

    learning_rate = config['learning_rate']

    lr_schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=1000,    # your batches_per_epoch
            decay_rate=0.97,
            staircase=True
    )

    optimizer = optax.chain( optax.clip_by_global_norm(1.0),
                             optax.adam(learning_rate=lr_schedule))

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    steps = config['steps']

    for step, (batch, y) in zip(range(steps), infinite_trainloader(trainloader)):

        main_key, sub_key, time_key = jax.random.split(main_key, 3)

        t = jax.random.choice(time_key, jnp.arange(1, config['trajectory_length']))
        
        loss, grads = image.compute_loss(model, sub_key,t, batch.numpy(), normalized_noise_level)

        pprint.pp(named_grad_norms(grads))

        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)

        print(f"loss for t={t} is {loss.item()}")

    save_model(config.model_save_name, model)


@cli.command()
@click.option("--seed", default=1900, help="seed.")
@click.option(
    "--config",
    default="./configs/training/spiral.yaml",
    help="Filename to save model checkpoint",
)
def spiral(seed, config):

    key = jax.random.PRNGKey(seed)

    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    data = spiral_.swissroll(config['data_count'], key)

    timesteps = config['trajectory_length']
    epochs    = config['epochs']
    learning_rate = config['learning_rate']

    betas = jnp.linspace(1e-4, 0.1, timesteps)

    _, diffusion_data = spiral_.forward_process(betas, data, key)

    fig, ax = plt.subplots(nrows=5, ncols=8, figsize=(15, 12))

    for i in range(5):
        for j in range(8):
            index = i * 5 + j
            ax[i, j].scatter(diffusion_data[index][:, 0], diffusion_data[index][:, 1])
    plt.savefig("plot.png")

    diffusion_data = jnp.permute_dims(diffusion_data, (1, 0, 2))  # B, T, 2

    optimizer = optax.adam(learning_rate)

    key, key_model, key_sample = jax.random.split(key, 3)

    model = spiral_.RBFNetwork(key=key_model)

    opt_state = optimizer.init(model)

    for i in range(epochs):
        y, x = diffusion_data[:, :-1, :], diffusion_data[:, 1:, :]

        loss, grads = spiral_.compute_loss(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)

        print(loss.item())

    save_model(config["model_save_name"], model)


if __name__ == '__main__':
    cli()
