# pyYBJ

Here we solve a 3D version of the YBJ equation [[1]](#1) pseudospectrally using Dedalus. We implement the horizontal dimensions as periodic (Foruier) and the vertical dimension as finite (Chebyshev). The mesoscale streamfunction is prescribed and can vary as a function of time. The background stratification profile is also prescribed but does not evolve in time. We augment the standard YBJ equation with a forcing term. The vertical structure of this forcing is presecribed.


## References
<a id="1">[1]</a> 
Young, W. R., Ben Jelloul, M. (1997).
Propagation of near-inertial oscillations through a geostrophic flow
Journal of Marine Research, 55(4), 735-766.
