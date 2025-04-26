import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array, Float, Integer


def simple_inference(
    diffusion_model: eqx.Module,
    key: jax.PRNGKeyArray,
    timesteps: int,
    noise_shape: Integer[Array, "..."],
    output_as_perturbation: bool = True,
) -> Float[Array, "batch timesteps channel spatial_width spatial_width "]:
    """

    Utter simple reverse sampling algorithm




    """

    key, data_key = jax.random.split(key, 2)

    x = random.normal(data_key, noise_shape)  # 1000,

    v_reverse_diffusion = jax.vmap(diffusion_model, in_axes=(None, 0))

    def f(acc, t):

        key, x = acc

        acc_key, key = jax.random.split(key, 2)

        mu, sigma = v_reverse_diffusion(t, x)

        if output_as_perturbation:
            # For the spiral dataset
            x += mu + random.normal(key, noise_shape) * sigma
        else:
            x = mu + random.normal(key, noise_shape) * sigma

        return (acc_key, x), x

    _, result = jax.lax.scan(f, (key, x), jnp.arange(timesteps - 1, -1, -1))

    return result
