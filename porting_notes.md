## Documentation for porting from the original code


### Reason for not using upscaling and downscaling in OG MultiScaleConvolution

https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models/blob/master/regression.py#L22

because `n_scale=1` makes downsream code like

https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models/blob/master/regression.py#L39

```python

for scale in range(self.num_scales):

```

evals to `[0]` and image_size (arguments) within the `conv` is reduced from `image_size=(spatial_width/2**scale, spatial_width/2**scale)` to `image_size=(spatial_width/2**0, spatial_width/2**0)` to `image_size=(spatial_width, spatial_width)`

and 

```python

for scale in range(self.num_scales-1, -1, -1):

   # downsample image to appropriate scale
   imgs_down = self.downsample(X, scale)

   ...

   if scale > 0:

      upscale code ...
```

evals to `[0]` as well and so it's easy to see how  upscaling and downsample is skipped 



### Reason for changing MLP DENSE UPPER TO CONV

https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models/blob/master/regression.py#L168

https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models/blob/master/regression.py#L189

The interesting thing about [Blocks](https://github.com/mila-iqia/blocks/tree/master) is that MLP are applied to the last axis of the input tensor (multiarray)

This is evidenced by the lack of a flattenting operation (^see second link above) before the input . This is counter to similar operations in modern deep learning frameworks

So in the original implementation the MLP was being applied to the features per pixel and to be ported made sense to make each of (~103 features post concatenation) a channel (filter)

in a convolution operation with a `kernel_size=1`
