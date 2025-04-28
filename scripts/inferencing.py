import click
import jax
import jaxtyping
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import equinox as eqx
import yaml

from sohl2015.image import Diffusion
from sohl2015.spiral import RBFNetwork
from utils import load_model, generate_denoising_animation, correct_image
from inference import simple_inference


@click.group()
def cli():
    """CLI for training or running inference."""
    pass


@cli.command()
@click.option("--seed", default=1900, help="seed.")
@click.option("--samples", default=10, help="number of images to draw")
@click.option(
    "--config",
    default="./configs/training/mnist.yaml",
    help="Config Filename",
)
def sohl(seed, samples, config):

    key = jax.random.PRNGKey(seed)

    model_key, sampling_key = jax.random.split(key, 2)

    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    spatial_width = config['spatial_width']
    n_colours=config['n_colours']
    trajectory_length = config['trajectory_length']
    
    model = Diffusion(
        model_key ,
        n_temporal_basis=config['n_temporal_basis'],
        spatial_width=spatial_width,
        n_hidden_dense_lower_output=config['n_hidden_dense_lower_output'],
        n_hidden_dense_lower=config['n_hidden_dense_lower'],
        n_hidden_dense_upper=config['n_hidden_dense_upper'],
        n_layers_conv=config['n_layers_conv'],
        n_layers_dense_lower=config['n_layers_dense_lower'],
        n_layers_dense_upper=config['n_layers_dense_upper'],
        n_colours=n_colours,
        n_hidden_conv=config['n_hidden_conv'],
        trajectory_length = trajectory_length 
    )

    model = load_model(model, f"./models/{config['model_save_name']}", key=model_key)

    samples = samples  # TODO: come back to figure out how to pass this in
    sampled = simple_inference(
        model,
        sampling_key,
        trajectory_length,
        (samples, n_colours, spatial_width, spatial_width),
        output_as_perturbation=False,
    )

    
    for i in range(samples):

        fix, ax = plt.subplots(figsize=(15, 15))
        im = ax.imshow(correct_image(sampled[-1, i].transpose(1, 2, 0)))
        ax.axis("off")
        plt.savefig(f"/tmp/denoised_cifar_{i}.png")

        generate_denoising_animation(f"./samples/new_denoised_cifar_{i}.gif", sampled[:, i], fps=60)

    return


@cli.command()
@click.option("--seed", default=1900, help="seed.")
@click.option(
    "--config",
    default="./configs/training/spiral.yaml",
    help="Parameter config",
)
@click.option(
    "--filename",
    default="./models/spiral.epx",
    help="Filename to save model checkpoint",
)
def spiral(seed, config, filename):

    key = jax.random.PRNGKey(seed)

    passon_key, model_key = jax.random.split(key, 2)

    model = RBFNetwork(key=model_key)

    model = load_model(model, filename, key=model_key)

    # ---------------------------------------------------------------------------

    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    trajectory_length = config['trajectory_length']

    samples = 1000  # TODO: come back to figure out how to pass this in
    sampled = simple_inference(
        model, passon_key, trajectory_length, (samples, 2), output_as_perturbation=True
    )

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    scat = ax.scatter(sampled[-1, :, 0], sampled[-1, :, 1])

    plt.savefig("/tmp/denoised_spiral.png")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.axis("off")

    scat = ax.scatter(sampled[0, :, 0], sampled[0, :, 1])

    def animate(i):
        data = [(x[0], x[1]) for x in sampled[i]]
        scat.set_offsets(data)

        return (scat,)

    ani = animation.FuncAnimation(
        fig,
        animate,
        repeat=True,
        frames=trajectory_length - 1,
        interval=50,
        repeat_delay=2500,
    )

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(
        fps=7, metadata=dict(artist="metric-space"), bitrate=1800
    )

    ani.save("/tmp/scatter.gif", writer=writer)

    return


if __name__ == "__main__":

    cli()
