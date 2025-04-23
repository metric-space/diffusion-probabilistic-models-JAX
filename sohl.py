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
import functools as ft
from jax.tree_util import tree_structure, tree_flatten, tree_flatten_with_path


# delete after
from PIL import Image

# --------------------- Network ----------------------------------------


def LeakyRelu(x):
    return jax.nn.leaky_relu(x, negative_slope=0.05)


# NOTE: num filters is the same as out channels
# border="full" implies padding should be done to the input which is why the crop logic is in place
class SingleScaleConvolution(eqx.Module):
    conv: eqx.nn.Conv
    spatial_width: int = eqx.static_field()

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

    spatial_width: int = eqx.static_field()
    colours: int = eqx.static_field()
    n_hidden_dense_lower_output: int = eqx.static_field()
    n_hidden_conv: int = eqx.static_field()
    n_temporal_basis: int = eqx.static_field()

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
class Diffusion(eqx.Module):
    temporal_basis: jax.Array
    beta_arr: jax.Array
    mlpconv: eqx.Module

    trajectory_length: int = eqx.static_field()
    n_temporal_basis: int = eqx.static_field()
    spatial_width: int = eqx.static_field()
    n_colours: int = eqx.static_field()

    def __init__(
        self,
        key,
        spatial_width,
        n_colours,
        trajectory_length=1000,
        n_temporal_basis=10,
        n_hidden_dense_lower=500,
        n_hidden_dense_lower_output=2,
        n_hidden_dense_upper=20,
        n_hidden_conv=20,
        n_layers_conv=4,
        n_layers_dense_lower=4,
        n_layers_dense_upper=2,
        step1_beta=0.001,
    ):

        key_beta, key_mlpconv = jax.random.split(key,2)


        self.trajectory_length = trajectory_length
        self.n_temporal_basis = n_temporal_basis
        self.spatial_width = spatial_width
        self.n_colours = n_colours

        self.temporal_basis = self.generate_temporal_basis(
            trajectory_length, n_temporal_basis
        )
        min_beta = self.generate_min_beta(trajectory_length, step1_beta)

        beta_perturb_coefficients = jax.random.normal(key_beta, (n_temporal_basis,))
        self.beta_arr = self.generate_beta_arr(min_beta, beta_perturb_coefficients)

        

        self.mlpconv =  MLPConvDense(
                key_mlpconv,
                n_temporal_basis=n_temporal_basis,
                spatial_width=spatial_width,
                n_hidden_dense_lower_output=n_hidden_dense_lower_output,
                n_hidden_dense_lower=n_hidden_dense_lower,
                n_hidden_dense_upper=n_hidden_dense_upper,
                n_layers_conv=n_layers_conv,
                n_layers_dense_lower=n_layers_dense_lower,
                n_layers_dense_upper=n_layers_dense_upper,
                n_colours=n_colours,
                n_hidden_conv=n_hidden_conv
        )

    def get_t_weights(self, t):
        # output: [trajectory_length, 1]
        t_compare = jnp.arange(self.trajectory_length)
        diff = jnp.abs(t_compare - t)
        weights = jnp.maximum(0.0, 1.0 - diff)
        return weights[:, None]

    def generate_temporal_basis(self, trajectory_length, n_basis):

        temporal_basis = jnp.zeros((trajectory_length, n_basis))

        xx = jnp.linspace(-1, 1, trajectory_length)  # [trajectory_length, :]

        x_centers = jnp.linspace(-1, 1, n_basis)

        width = (x_centers[1] - x_centers[0]) / 2.0

        temporal_basis = jnp.exp(
            -((xx[:, None] - x_centers[None, :]) ** 2) / (2 * width**2)
        )

        print(f"temporal_basis shape is {temporal_basis.shape}")

        temporal_basis /= jnp.sum(temporal_basis, axis=1).reshape((-1, 1))
        temporal_basis = temporal_basis.T

        print(f"temporal_basis shape is {temporal_basis.shape}")

        return temporal_basis  # [self.n_basis, trajectory_length]

    def temporal_readout(self, Z, t):

        Z = Z.reshape((self.spatial_width, self.spatial_width, self.n_colours, 2, self.n_temporal_basis))

        t_weights = self.get_t_weights(t)

        coeff_weights = self.temporal_basis @ t_weights  # [10,1]

        concat_coeffs = Z @ coeff_weights

        print(f"concat_coeffs weights are: {concat_coeffs.shape}")

        mu_coeff = jnp.permute_dims(concat_coeffs[:, :, :, 0], (2, 0, 1, 3)).squeeze(-1)

        sigma_coeff = jnp.permute_dims(concat_coeffs[:, :, :, 1], (2, 0, 1, 3)).squeeze(
            -1
        )

        return mu_coeff, sigma_coeff

    def generate_min_beta(self, trajectory_length, step1_beta):
        min_beta_val = 1e-6
        min_beta_values = jnp.ones((trajectory_length,)) * min_beta_val
        min_beta_values += jnp.eye(trajectory_length)[0, :] * step1_beta

        return min_beta_values

    def generate_beta_arr(self, min_beta, beta_perturb_coefficients):

        beta_perturb = self.temporal_basis.T @ beta_perturb_coefficients

        beta_baseline = 1.0 / jnp.linspace(self.trajectory_length, 2.0, self.trajectory_length)
        beta_baseline_offset = jax.scipy.special.logit(beta_baseline)

        beta_arr = jax.nn.sigmoid(beta_perturb + beta_baseline)
        beta_arr = min_beta + beta_arr * (1 - min_beta - 1e-5)

        return beta_arr.reshape((self.trajectory_length, 1))

    def get_beta_forward(self, t):

        return self.get_t_weights(t).T @ self.beta_arr

    # TODO: batch yet to come
    def forward_diffusion(self, key, t,  image):

        # Remember to split key
        choice_key, normal_key, uniform_key = jax.random.split(key,3) 


        #t = jax.random.choice(choice_key, jnp.arange(1, self.trajectory_length))
        t_weights = self.get_t_weights(t)

        N = jax.random.normal(normal_key, image.shape)
        U = jax.random.uniform(uniform_key, image.shape)

        beta_forward = self.get_beta_forward(t)
        alpha_forward = 1 - beta_forward

        alpha_arr = 1 - self.beta_arr
        alpha_cum_forward_arr = jnp.cumprod(alpha_arr)
        alpha_cum_forward = t_weights.T @ alpha_cum_forward_arr

        image_uniform_noise = image + U 
        # NOTE: the shape
        image_noise = image_uniform_noise * jnp.power(
            alpha_cum_forward, 0.5
        ) + N * jnp.power(1 - alpha_cum_forward, 0.5)

        # NOTE: figure why this even works
        mu1_sc1 = jnp.power(alpha_cum_forward / alpha_forward, 0.5)
        mu2_sc1 = 1.0 / jnp.power(alpha_forward, 0.5)
        cov1 = 1.0 - alpha_cum_forward / alpha_forward
        cov2 = beta_forward / alpha_forward
        lam = 1.0 / cov1 + 1.0 / cov2
        mu = (
            (image_uniform_noise * mu1_sc1 / cov1) + (image_noise * mu2_sc1 / cov2)
        ) / lam
        sigma = jnp.power(lam, -0.5).reshape((1, 1, 1))  # huh?

        return image_noise, mu, sigma

    # TODO: name this better
    def reverse_diffusion(self, t, noised_image):

        Z = self.mlpconv(noised_image)
        mu_coeff, beta_coeff = self.temporal_readout(Z, t)

        # reverse variance is a perturbation around forward variance  (why?)
        beta_forward = self.get_beta_forward(t)

        beta_coeff_scaled = beta_coeff / jnp.power(self.trajectory_length, 0.5)
        beta_reverse = jax.nn.sigmoid(beta_coeff_scaled + jax.scipy.special.logit(beta_forward))

        sigma = jnp.power(beta_reverse, 0.5)

        mu = noised_image * jnp.power(1.0 - beta_forward, 0.5) + mu_coeff * jnp.power(
            beta_forward, 0.5
        )

        return mu, sigma



def neg_loglikelihood(
    mu,
    sigma,
    mu_posterior,
    sigma_posterior,
    trajectory_length,
    covariance_schedule,
    n_colours=3,
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

    model = Diffusion(
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
        trajectory_length = 1000
    )

    @eqx.filter_value_and_grad
    def compute_loss(key, t, model_, images):
        keys = jax.random.split(key, 100)

        v_forward_diffusion = jax.vmap(model_.forward_diffusion, in_axes=(0,None,0))
        v_reverse_diffusion = jax.vmap(model_.reverse_diffusion, in_axes=(None,0))

        image, mu, sigma = v_forward_diffusion(keys, t, images)
        r_mu, r_sigma = v_reverse_diffusion(t, image)
        print(f"Shapes: {image.shape}, {mu.shape}, {sigma.shape}, {r_mu.shape}, {sigma.shape}")
        loss =  neg_loglikelihood(r_mu, r_sigma, mu, sigma, 1000, model_.beta_arr, 3)

        return model_.beta_arr.mean()

    
    v_neg_loglikelihood = jax.vmap(neg_loglikelihood, in_axes=(0,0,0,0,None,None,None))

    t = jax.random.choice(main_key, jnp.arange(1, 1000)) 

    learning_rate = 1e-3

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # ---- debug code ------------------
    params = eqx.filter(model, eqx.is_inexact_array)
    flat = tree_flatten_with_path(params)

    for path, leaf in flat[0]:
        print(".".join(str(p) for p in path), "-->", leaf.shape if hasattr(leaf, "shape") else "scalar",  leaf.dtype if getattr(leaf, 'dtype', None) else " No dtype")

    leaves, _ = tree_flatten(opt_state)
    

    for batch in trainloader:
        
        print(batch[0].shape)

        images = batch[0].numpy()

        loss, grads = compute_loss(main_key,t,  model, images)
        
        print(loss.item())
        print(grads)

        print("GRAD STRUCTURE")
        print(jax.tree_util.tree_structure(grads))

        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)

        print(loss.item())

        break

    print("END")

    print(f"mu shape is {mu.shape} and sigma shape is {sigma.shape}")
