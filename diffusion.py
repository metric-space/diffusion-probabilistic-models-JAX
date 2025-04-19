import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import click
import os

import matplotlib.animation as animation


# jax typing?


#  dataset
def swissroll(n, key):
    # Swiss roll dataset with zero mean, unit sd

    key_phi, key_noise = jax.random.split(key, 2)

    constant = 1.2

    phi = (jax.random.uniform(key_phi, (n,)) * 2.6 + constant) * jnp.pi

    x, y = jnp.cos(phi), jnp.sin(phi)
    data = phi[:, None] * jnp.stack([x, y], axis=1)

    noise = jax.random.uniform(key_noise, (n, 2)) * 0.01

    data = data + noise
    data -= data.mean(axis=0)
    data /= data.std()
    return data


# variance here refers to beta
# TODO: think about key placement
def forward_process_single_step(acc, std):
    key, data = acc
    # assert shape here
    key_, key_noise = jax.random.split(key, 2)

    # split key here

    noise = jax.random.normal(key_noise, data.shape) * std

    noised_data = ((1 - std) ** 0.5) * data + noise

    return (key_, noised_data), noised_data


# iterate over steps zipped with


# 1 trajectory step
def forward_process(beta, init_data, key, trajectories=15):

    init_data = init_data.repeat(repeats=trajectories, axis=0)

    return jax.lax.scan(forward_process_single_step, (key, init_data), beta)


#  NOTE: broadcasting kept out of function as the author's belief
# is that function should be kept ignorant (less magical/surprises)
def multivariate_normal_diag(x, mu, sigma, D=2, log=False):
    scaled_diff = (x - mu) / sigma
    expon = -0.5 * (scaled_diff**2).sum(axis=-1)

    if log:
        logdenomsq = D * jnp.log(2 * jnp.pi) + jnp.log(sigma.prod(axis=-1))
        lognorm = -0.5 * logdenomsq
        return lognorm + expon
    else:
        denomsq = (2 * jnp.pi) ** D * sigma.prod(axis=-1)
        norm = denomsq**-0.5
        return norm * jnp.exp(expon)


def pairwise_multivariate_normal_diag(x, mu, sigma):
    broadcast_x = jnp.expand_dims(x, -2)
    D = x.shape[-1]

    return multivariate_normal_diag(broadcast_x, mu, sigma, D)


class RBFNetwork(eqx.Module):
    center_params: jax.Array
    shape_params: jax.Array
    mu_params: jax.Array
    sigma_params: jax.Array

    def __init__(self, key, Hsqrt=4, D=2, T=39):
        centers_key, shapes_key, mu_key, sigma_key = jax.random.split(key, 4)

        H = Hsqrt**2

        pts = jnp.linspace(-2, 2, Hsqrt)

        self.center_params = jnp.stack(
            jnp.meshgrid(*2 * [pts], indexing="ij"), -1
        ).reshape(
            -1, 2
        )  # jax what a let down: all this crap for a lousy cartesian product
        self.shape_params = jax.random.normal(shapes_key, (H, D))
        self.mu_params = jax.random.normal(mu_key, (T, H, D))
        self.sigma_params = jax.random.normal(sigma_key, (T, H, D))

    def __call__(self, x, ts=None):
        sigma_params = self.sigma_params
        mu_params = self.mu_params

        if ts is not None:
            # x is B,2
            x = jnp.expand_dims(x, 1)  # why though?
            sigma_params = jnp.expand_dims(sigma_params[ts], 0)
            mu_params = jnp.expand_dims(mu_params[ts], 0)

        pdf = pairwise_multivariate_normal_diag(
            x, self.center_params, jax.nn.sigmoid(self.shape_params)
        )

        normalizer = 1.0 / pdf.sum(axis=-1, keepdims=True)  # b, T, 1

        pdf = jnp.expand_dims(pdf, axis=-1)  # B, T, H, 1

        mu = (pdf * mu_params).sum(axis=2) * normalizer  # B, T, 2

        sigma_logits = (pdf * sigma_params).sum(axis=2) * normalizer  # B, T, 2

        sigma = jax.nn.sigmoid(sigma_logits)

        if ts is not None:
            return mu.squeeze(1), sigma.squeeze(1)

        return mu, sigma


def reverse_sample(rbf_network_model, key, samples=1000, timesteps=39):
    # split key here

    keys_ = jax.random.split(key, timesteps + 1)

    x = jax.random.normal(keys_[0], (samples, 2))  # 1000,

    steps = [x]

    for t in reversed(range(timesteps)):
        mu, sigma = rbf_network_model(x, t)
        # split key here
        x = x + mu + jax.random.normal(keys_[t + 1], (samples, 2)) * sigma
        steps.append(x)

    print([x.shape for x in steps])

    return jnp.stack(steps)  # T, B, 2 perhaps we should change this?


@eqx.filter_value_and_grad
def compute_loss(model, x, y):
    mu, sigma = model(x)

    return -multivariate_normal_diag(y, x + mu, sigma, log=True).mean()


# NOTE: this will save to a directory in the root directory, not a file in the root directory
def save_model(filename, model):

    directory = "./models"

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), "wb") as f:
        eqx.tree_serialise_leaves(f, model)

def load_model(filename, key):
    with open(filename, "rb") as f:
        model = RBFNetwork(key=key)
        return eqx.tree_deserialise_leaves(f, model)



@click.group()
def cli():
    """CLI for training or running inference."""
    pass

@cli.command()
@click.option("--epochs", default=10000, help="Number of training epochs.")
@click.option("--lr", default=0.007, help="Learning rate.")
@click.option("--filename",default="spiral.epx",help="Filename to save model checkpoint")
def train(epochs, lr, filename):
    """Run training."""
    click.echo(f"Training for {epochs} epochs with lr={lr}")

    key = jax.random.PRNGKey(0)
    data = swissroll(1000, key)

    timesteps = 40

    betas = jnp.linspace(1e-4, 0.1, timesteps)

    _, diffusion_data = forward_process(betas, data, key)

    data = diffusion_data

    fig, ax = plt.subplots(nrows=5, ncols=8, figsize=(15, 12))

    for i in range(5):
        for j in range(8):
            index = i * 5 + j
            ax[i, j].scatter(data[index][:, 0], data[index][:, 1])
    plt.savefig("plot.png")

    # --------------------------------------------

    learning_rate = 0.007

    # -------------------------------------------

    data = jnp.permute_dims(data, (1, 0, 2))  # B, T, 2

    optimizer = optax.adam(learning_rate)

    key, key_model, key_sample = jax.random.split(key, 3)

    model = RBFNetwork(key=key_model)

    opt_state = optimizer.init(model)

    for i in range(epochs):
        y, x = data[:, :-1, :], data[:, 1:, :]

        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)

        print(loss.item())

    save_model(filename, model)


@cli.command()
@click.option("--checkpoint", required=True, help="Path to model checkpoint.")
def inference(checkpoint):
    """Run inference."""

    key = jax.random.PRNGKey(0)

    key_model, key_sample = jax.random.split(key)

    model = load_model(checkpoint, key)

    click.echo(f"Running inference with checkpoint={checkpoint}")
     
    sampled = reverse_sample(model, key_sample)

    print("plotting")

    #fig, ax = plt.subplots(nrows=5, ncols=8, figsize=(15, 12))

    #for i in range(5):
    #    for j in range(8):
    #        index = i * 8 + j
    #        print(index)
    #        ax[i, j].scatter(sampled[index][:, 0], sampled[index][:, 1])

    #plt.savefig("final_with_trajectories.png")

    #fix, ax = plt.subplots(figsize=(15,15))
    #ax.scatter(sampled[-1][:,0], sampled[-1][:,1])
    #plt.savefig("denoised.png")


    fig, ax = plt.subplots(figsize=(15,15))
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    print(sampled[0][:,0][0], sampled[0][:,1][1])

    scat = ax.scatter(sampled[0][:,0], sampled[0][:,1])

    x = jnp.linspace(0.2, 1, 40)
    print(x)

    def animate(i):
         if i > 0 :
            data = [(x[0], x[1]) for x in sampled[i]]
            print(data)
            scat.set_offsets(data)
         else:
             pass
            
         print(f"{i}th pass is somewhat successful")
         return scat,

    ani = animation.FuncAnimation(fig, animate, repeat=True, frames=40 - 1, interval=50, repeat_delay=2500)
    
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=7,
                                    metadata=dict(artist='metric-space'),
                                    bitrate=1800)
    ani.save('scatter.gif', writer=writer)
    #plt.show()



if __name__ == "__main__":
    cli()
    

    


