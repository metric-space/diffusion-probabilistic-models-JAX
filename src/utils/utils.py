import equinox as eqx 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def save_model(filename, model, directory="./models"):

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load_model(model, filename, key):
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model)


def generate_denoising_animation(gif_filename, trajectory, figsize=(4,4), fps=7, repeat_delay=2500):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(trajectory[0].transpose(1, 2, 0))  # Convert (C, H, W) → (H, W, C)
    ax.axis('off')
    
    # Animation function
    def animate(i):
        im.set_data(trajectory[i].transpose(1, 2, 0))
        return im,

    timesteps = trajectory.shape[0]
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, animate, frames=timesteps, interval=100, blit=True, repeat=True, repeat_delay=repeat_delay
    )
    
    # Save as GIF
    writer = animation.PillowWriter(fps=fps, metadata=dict(artist='metric-space'))
    ani.save(gif_filename, writer=writer)
