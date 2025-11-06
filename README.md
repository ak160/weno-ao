# weno-ao

# Problem Statement : 
Traditional image upscaling methods, like bicubic interpolation, suffer from a poor trade-off between sharpness and smoothness, often producing blurring and ringing artifacts near edges.
While advanced standard WENO schemes were designed to capture sharp edges without these oscillations , they suffer from a critical limitation: order reduction at critical points. Since these local minima and maxima mathematically define an image's fine textures, this failure causes standard WENO to blur and flatten essential details.
The core problem is therefore the lack of a method that can simultaneously provide robust stability at sharp edges and maintain the high-order accuracy needed for fine textures. This necessitates an adaptive-order (AO) approach, like the WENO-AO(5,3) scheme, which can dynamically adjust its accuracy to solve both problems at once.
