import click
import jax
import jaxtyping
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import equinox as eqx

from sohl2015.image import Diffusion
from sohl2015.spiral import RBFNetwork
from utils import load_model, generate_denoising_animation
from inference import simple_inference
from dataclasses import dataclass


@dataclass
class Config:
    model: eqx.Module
    shape: tuple[int]
    key: jaxtyping.PRNGKeyArray
    trajectory: int



@click.group()
def cli():
    """CLI for training or running inference."""
    pass


@cli.command()
@click.option("--seed", default=1900, help="seed.")
@click.option("--filename",default="./models/mnist_better_hyparameters.ex",help="Filename to save model checkpoint")
def sohl(seed, filename):

    key = jax.random.PRNGKey(seed)

    passon_key, model_key = jax.random.split(key, 3)

    n_hidden_dense_lower_output = 5
    n_hidden_dense_lower = 1000
    n_hidden_dense_upper = 100
    n_hidden_conv = 100
    n_layers_conv = 6
    n_layers_dense_lower = 6
    n_layers_dense_upper = 4

    spatial_width = 28
    n_temporal_basis = 20

    trajectory_length = 700

    model = Diffusion(
        model_key ,
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

    model = load_model(model, filename, key=model_key)

    # TODO: come back to the sample
    return Config(model=model, key=passon_key, trajectory=700, shape=(10,1,spatial_width,spatial_width))


@cli.command()
@click.option("--seed", default=1900, help="seed.")
@click.option("--filename",default="./models/spiral.epx",help="Filename to save model checkpoint")
def spiral(seed, filename):

    key = jax.random.PRNGKey(seed)

    passon_key, model_key = jax.random.split(key, 2)

    model = RBFNetwork(key=model_key)

    model = load_model(model, filename, key=model_key)


    # ---------------------------------------------------------------------------

    trajectory_length = 40

    samples = 1000   # TODO: come back to figure out how to pass this in
    sampled = simple_inference(model, passon_key, trajectory_length, (samples, 2), output_as_perturbation=True)


    fig, ax = plt.subplots(figsize=(15,15))
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    scat = ax.scatter(sampled[-1, :, 0], sampled[-1, : ,1])

    plt.savefig("./denoised_spiral.png")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.axis('off')

    scat = ax.scatter(sampled[0, :, 0], sampled[0, :, 1])
    def animate(i):
        data = [(x[0], x[1]) for x in sampled[i]]
        scat.set_offsets(data)

        return scat,

    ani = animation.FuncAnimation(fig, animate, repeat=True, frames=trajectory_length-1, interval=50, repeat_delay=2500)
    
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=7,
                                    metadata=dict(artist='metric-space'),
                                    bitrate=1800)

    ani.save('scatter.gif', writer=writer)

     
    return Config(model=model, key=passon_key, trajectory=40 , shape=(10,2))




if __name__ == '__main__':

    config = cli()
    print(config)

    samples = 10   # TODO: come back to figure out how to pass this in
    sampled = simple_inference(model, sampling_key, trajectory_length, (samples, 2), output_as_perturbation=True)

    fix, ax = plt.subplots(figsize=(15,15))
    im = ax.imshow(sampled[-1,0].transpose(1, 2, 0))
    ax.axis('off')
    plt.savefig("denoised_spiral.png")

    print(sampled.shape)

    #for i in range(samples):
    generate_denoising_animation(f"/tmp/new_denoised_spiral.gif", sampled[:,0], fps=60)
