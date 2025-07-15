# low-framerate to high-framerate video generation
Temporal upsampling of videos is done using a paper developed by Fitsum Reda et al ([FILM: Frame Interpolation for Large Motion
](https://arxiv.org/abs/2202.04901)). The training and inference code was adapted from this and ["Video to Events: Recylcing Video Datasets for Event Cameras"](https://arxiv.org/abs/1912.03095).

The input to the network is the following video :
<video controls preload="metadata" style="width:100%">
  <source src="original_h264.mp4"  type="video/mp4">
</video>

And the result of that can be seen below : 
<video controls preload="metadata" style="width:100%">
  <source src="upsampling_h264.mp4"  type="video/mp4">
</video>

# Examples


# Network