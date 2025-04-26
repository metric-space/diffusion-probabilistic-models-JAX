import jax
import jax.random as random
import jax.numpy as jnp


def simple_inference(diffusion_model, key, timesteps, noise_shape, output_as_perturbation=True):
    """

     


    """

    keys_ = jax.random.split(key, timesteps + 1)

    x = jax.random.normal(keys_[0], noise_shape)  # 1000,

    steps = [x]

    v_reverse_diffusion = jax.vmap(diffusion_model, in_axes=(None,0))

    for t in reversed(range(timesteps)):
        mu, sigma = v_reverse_diffusion(t,x)
        # split key here
        if output_as_perturbation:
            x += mu + random.normal(keys_[t + 1], noise_shape) * sigma
        else:
            x = mu + random.normal(keys_[t + 1], noise_shape) * sigma
        steps.append(x)

    print([x.shape for x in steps])

    return jnp.stack(steps)  # T, B, 2 perhaps we should change this?
