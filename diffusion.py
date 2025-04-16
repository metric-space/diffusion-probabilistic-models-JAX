import jax
import jax.numpy as jnp
import jax.lax as lax
import matplotlib.pyplot as plt
# jax typing?


#  dataset
def swissroll(n, key):
    # Swiss roll dataset with zero mean, unit sd

    key_phi, key_noise = jax.random.split(key, 2)

    constant = 1.2

    phi = (jax.random.uniform(key_phi,(n,)) * 2.6 + constant)*jnp.pi

    x, y = jnp.cos(phi),  jnp.sin(phi)
    data = phi[:, None] * jnp.stack([x,y],axis=1)

    noise = jax.random.uniform(key_noise,(n,2)) * 0.01

    data = data + noise
    data -= data.mean(axis=0)
    data /= data.std()
    return data


# variance here refers to beta
# TODO: think about key placement
def forward_process_single_step(acc, std):

    key, data = acc
    # assert shape here
    key_, key_noise = jax.random.split(key,2) 

    # split key here

    noise = jax.random.normal(key_noise, data.shape) * std

    noised_data = ((1-std**(0.5))*data + noise)

    return (key_, noised_data), noised_data


# iterate over steps zipped with


# 1 trajectory step
def forward_process(beta, init_data, key):

    # data = data[None,:].repeat(trajectory_steps, axis=0)

    return jax.lax.scan(forward_process_single_step,(key, init_data), beta)




# Iterate 





if __name__ == '__main__':

    key = jax.random.PRNGKey(0)
    data = swissroll(1000, key)

    timesteps = 40

    betas = jnp.linspace(1e-4, 0.1, timesteps)

    _, diffusion_data = forward_process(betas, data ,key)

    data = diffusion_data


    fig, ax = plt.subplots(nrows=5, ncols=8, figsize=(15, 12))

    for i in range(5):
        for j in range(8):
            index = i*5+j
            ax[i,j].scatter(data[index][:,0], data[index][:,1])
    plt.savefig("plot.png")





