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


# delete after
from PIL import Image

# --------------------- Network ----------------------------------------


def LeakyRelu(x):
    return jax.nn.leaky_relu(x, negative_slope=0.05)


# NOTE: num filters is the same as out channels
# border="full" implies padding should be done to the input which is why the crop logic is in place
class SingleScaleConvolution(eqx.Module):
    layers: list
    spatial_width: int

    def __init__(self, key, num_channels, num_filters, spatial_width, filter_size):
        self.spatial_width = spatial_width
        self.layers = [
                eqx.nn.Conv(key=key, num_spatial_dims=2,in_channels=num_channels, out_channels=num_filters,  kernel_size=filter_size, stride=1, padding=1),
                LeakyRelu
        ]

    # TODO: replace this with a library function, possibly from equinox
    # USELESS for now
    def downsample(self,x):
        # assuming shape 
        B,C,W,H = x.shape
        x = x.reshape((B, C, W // 2, 2, H // 2, 2))
        x = x.mean(axis=5)
        x = x.mean(axis=3)
        return x

    # TODO: replace this with something more easier to figure out
    # USELESS FOR NOW
    def upsample(self, x):
        C,W,H = x.shape
        x = x.reshape(( C, W, 1, H, 1))
        x = jnp.concat([x,x], axis=4)
        x = jnp.concat([x,x], axis=2)
        x = x.reshape((C, self.spatial_width, self.spatial_width))
        return x


    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x






# ----------------------------------------------------------------------

train_dataset = torchvision.datasets.CIFAR10(
    "CIFAR10",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=100, shuffle=True, drop_last=True
)


count = 2

import numpy as np

#for i in trainloader:
#    if count > 0:
#        print(type(i))
#        print(len(i))
#        print(i[0].shape)
#
#    count -= 1


arr = train_dataset[0][0].numpy()

print(arr.shape)

#arr = arr.reshape((C, W//2 , 2 , H // 2 , 2))
#arr = arr.mean(dim=4)
#arr = arr.mean(dim=2)

#print(arr.shape)

#arr_uint8 = (arr * 255).astype(np.uint8)

# Create and save image
# img = Image.fromarray(arr_uint8)
# img.save("output.png")

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    k, main_key = jax.random.split(key,2)

    c = SingleScaleConvolution(main_key, 3, 100, 32, 3)

    k = c(arr)

    print("k shape is ", k.shape)

