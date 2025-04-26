import jax
import matplotlib.pyplot as plt

from sohl2015.image import Diffusion
from utils import load_model, generate_denoising_animation
from inference import simple_inference



if __name__ == '__main__':


    key = jax.random.PRNGKey(39)

    initial_key, model_key , sampling_key = jax.random.split(key, 3)

    n_hidden_dense_lower_output = 5
    n_hidden_dense_lower = 1000
    n_hidden_dense_upper = 100
    n_hidden_conv = 100
    n_layers_conv = 6
    n_layers_dense_lower = 6
    n_layers_dense_upper = 4

    spatial_width = 28
    n_temporal_basis = 10

    trajectory_length = 300

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

    filename = "./models/sohl_cifar.ex"
    model = load_model(model, filename, key=model_key)
    sampled = simple_inference(model, sampling_key, 300, (1,1,28,28), output_as_perturbation=False)

    fix, ax = plt.subplots(figsize=(15,15))
    im = ax.imshow(sampled[-1,0].transpose(1, 2, 0))
    ax.axis('off')
    plt.savefig("denoised_mnist_new.png")

    print(sampled.shape)

    generate_denoising_animation("denoising_mnist_new.gif", sampled[:,0], fps=50)





