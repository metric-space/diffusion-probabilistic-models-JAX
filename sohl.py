import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import click
import os
import torchvision
import torch
import torchvision.transforms as transforms
from dataclasses import dataclass


# delete after
from PIL import Image

# --------------------- Network ----------------------------------------


def LeakyRelu(x):
    return jax.nn.leaky_relu(x, negative_slope=0.05)


# NOTE: num filters is the same as out channels
# border="full" implies padding should be done to the input which is why the crop logic is in place
class SingleScaleConvolution(eqx.Module):
    conv: eqx.nn.Conv
    spatial_width: int

    def __init__(self, key, num_channels, num_filters, spatial_width, filter_size):
        self.spatial_width = spatial_width
        self.conv = eqx.nn.Conv(
            key=key,
            num_spatial_dims=2,
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=1,
            padding=1,
        )

    # NOTE: dummy key is required to be used as a part of equinox.nn.Sequential
    def __call__(self, x, key=None):
        # pad input for 'full effect', remember input has no batch axis yet
        x = jax.numpy.pad(x, ((0, 0), (1, 1), (1, 1)))
        x = self.conv(x)
        x = x[:, 1:-1, 1:-1]
        x = LeakyRelu(x)

        return x


# TODO: change filter to kernel, it's 2025 not 2015
class MultiLayerConvolution(eqx.Module):
    layers: list  # or sequential?

    def __init__(
        self, key, n_layers, n_hidden, spatial_width, n_colours, filter_size=3
    ):

        self.layers = []
        # split keys here

        keys_ = jax.random.split(key, n_layers + 1)

        in_channels = n_colours

        for i in range(n_layers):
            self.layers.append(
                SingleScaleConvolution(
                    keys_[i + 1], in_channels, n_hidden, spatial_width, filter_size
                )
            )
            in_channels = n_hidden

        self.layers = eqx.nn.Sequential(self.layers)

    def __call__(self, x):
        return self.layers(x)


class MLPConvDense(eqx.Module):
    mlp_dense_upper: eqx.nn.MLP
    mlp_dense_lower: eqx.nn.MLP
    conv: MultiLayerConvolution

    spatial_width: int
    colours: int
    n_hidden_dense_lower_output: int
    n_hidden_conv: int
    n_temporal_basis: int

    def __init__(
        self,
        key,
        n_layers_conv,
        n_layers_dense_lower,
        n_layers_dense_upper,
        n_hidden_conv,
        n_hidden_dense_lower,
        n_hidden_dense_lower_output,
        n_hidden_dense_upper,
        spatial_width,
        n_colours,
        n_temporal_basis,
    ):

        _, key_conv, key_lower, key_upper = jax.random.split(key, 4)

        self.conv = MultiLayerConvolution(
            key_conv, n_layers_conv, n_hidden_conv, spatial_width, n_colours
        )

        input_ = n_colours * spatial_width**2
        output_ = n_hidden_dense_lower_output * spatial_width**2
        self.mlp_dense_lower = eqx.nn.MLP(
            key=key_lower,
            activation=LeakyRelu,
            depth=n_layers_dense_lower,
            in_size=input_,
            out_size=output_,
            width_size=n_hidden_dense_lower,
        )

        input_ = n_hidden_conv + 3  # n_hidden_dense_lower_output
        output_ = n_colours * n_temporal_basis * 2
        self.mlp_dense_upper = eqx.nn.Conv(
            key=key_upper,
            num_spatial_dims=2,
            in_channels=input_,
            out_channels=output_,
            kernel_size=1,
        )

        self.spatial_width = spatial_width
        self.colours = n_colours
        self.n_hidden_conv = n_hidden_conv
        self.n_hidden_dense_lower_output = n_hidden_dense_lower_output
        self.n_temporal_basis = n_temporal_basis

    def __call__(self, x):

        Y = self.conv(x)
        Y = jnp.permute_dims(Y, (1, 2, 0))

        X = x.reshape((self.colours * self.spatial_width**2))
        Y_dense = self.mlp_dense_lower(X)
        print(f"Y shape is {Y.shape}")
        Y_dense = X.reshape((self.spatial_width, self.spatial_width, 3))

        Z = jnp.concat(
            [
                Y / jnp.sqrt(self.n_hidden_conv),
                Y_dense / jnp.sqrt(self.n_hidden_dense_lower_output),
            ],
            axis=-1,
        )

        Z = jnp.permute_dims(Z, (2, 0, 1))

        print(f"Z shape is {Z.shape}")

        Z = self.mlp_dense_upper(Z)

        print(f"Z shape is {Z.shape}")

        Z = jnp.permute_dims(Z, (1, 2, 0))

        return Z


# ----------------------------------------------------------------------

train_dataset = torchvision.datasets.CIFAR10(
    "CIFAR10", train=True, download=True, transform=transforms.ToTensor()
)

trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=100, shuffle=True, drop_last=True
)


count = 2

import numpy as np

# for i in trainloader:
#    if count > 0:
#        print(type(i))
#        print(len(i))
#        print(i[0].shape)
#
#    count -= 1


arr = train_dataset[0][0].numpy()

print(arr.shape)

# arr = arr.reshape((C, W//2 , 2 , H // 2 , 2))
# arr = arr.mean(dim=4)
# arr = arr.mean(dim=2)

# print(arr.shape)

# arr_uint8 = (arr * 255).astype(np.uint8)

# Create and save image
# img = Image.fromarray(arr_uint8)
# img.save("output.png")


# TODO: make this a config

# TODO: change name
#class Diffusion(eqx.Module):
#    temporal_basis: None
#
#
#    def __init__(self, key):
#        pass
#
#    def get_t_weights(self, t):
#        # output: [trajectory_length, 1]
#        t_compare = jnp.arange(self.trajectory_length)
#        diff = jnp.abs(t_compare - t)
#        weights = jnp.maximum(0.0, 1.0 - diff)
#        return weights[:, None]
#
#    def generate_temporal_basis(self): #, trajectory_length, n_basis):
#
#        temporal_basis = jnp.zeros((self.trajectory_length, self.n_basis))
#
#        xx = jnp.linspace(-1,1,self.trajectory_length)
#
#        x_centers = jnp.linspace(-1,1, self.n_basis)
#
#        width = (x_centers[1] - x_centers[0])/2.
#
#        for i in range(self.n_basis):
#            temporal_basis[:,i] = jnp.exp(-(xx - x_centers[i])**2 / (2*width**2))
#        
#        temporal_basis /= jnp.sum(temporal_basis, axis=1)
#        temporal_basis = temporal_basis.T
#
#        return temporal_basis # [self.n_basis, trajectory_length]
#


def get_t_weights(trajectory_length, t):
    # output: [trajectory_length, 1]
    t_compare = jnp.arange(trajectory_length)
    diff = jnp.abs(t_compare - t)
    weights = jnp.maximum(0.0, 1.0 - diff)
    return weights[:, None]


def generate_temporal_basis(trajectory_length, n_basis):

    temporal_basis = jnp.zeros((trajectory_length, n_basis))

    xx = jnp.linspace(-1,1,trajectory_length) # [trajectory_length, :]

    x_centers = jnp.linspace(-1,1,n_basis)

    width = (x_centers[1] - x_centers[0])/2.

    temporal_basis = jnp.exp(-(xx[:,None] - x_centers[None,:])**2 / (2*width**2))

    print(f"temporal_basis shape is {temporal_basis.shape}")
    
    temporal_basis /= jnp.sum(temporal_basis, axis=1).reshape((-1,1))
    temporal_basis = temporal_basis.T

    print(f"temporal_basis shape is {temporal_basis.shape}")

    return temporal_basis # [self.n_basis, trajectory_length]


def generate_temporal_readout(trajectory_length, Z, t):

    Z = Z.reshape((32, 32, 3, 2, 10))

    temp_basis =  generate_temporal_basis(trajectory_length, 10)

    t_weights = get_t_weights(trajectory_length, t)

    coeff_weights = temp_basis @ t_weights # [10,1]

    concat_coeffs = Z @ coeff_weights

    print(f"concat_coeffs weights are: {concat_coeffs.shape}")

    mu_coeff = jnp.permute_dims(concat_coeffs[:,:,:,0], (2,0,1,3)).squeeze(-1)

    sigma_coeff = jnp.permute_dims(concat_coeffs[:,:,:,1], (2,0,1,3)).squeeze(-1)

    return mu_coeff, sigma_coeff



if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    k, main_key = jax.random.split(key, 2)

    n_hidden_dense_lower_output = 5
    n_hidden_dense_lower = 1000
    n_hidden_dense_upper = 100
    n_layers_conv = 100
    n_layers_dense_lower = 6
    n_layers_dense_upper = 100

    spatial_width = 32
    n_temporal_basis = 10

    # c = SingleScaleConvolution(main_key, 3, 100, 32, 3)

    # c = MultiLayerConvolution(main_key, 00, 32, 3)

    # k = c(arr)

    mlpconv = MLPConvDense(
        main_key,
        n_temporal_basis=n_temporal_basis,
        spatial_width=spatial_width,
        n_hidden_dense_lower_output=n_hidden_dense_lower_output,
        n_hidden_dense_lower=n_hidden_dense_lower,
        n_hidden_dense_upper=n_hidden_dense_upper,
        n_layers_conv=n_layers_conv,
        n_layers_dense_lower=n_layers_dense_lower,
        n_layers_dense_upper=n_layers_dense_upper,
        n_colours=3,
        n_hidden_conv=100,
    )

    image_ = mlpconv(arr)

    print("k shape is ", image_.shape)

    mu,sigma = generate_temporal_readout(100, image_, n_temporal_basis)

    print(f"mu shape is {mu.shape} and sigma shape is {sigma.shape}")
