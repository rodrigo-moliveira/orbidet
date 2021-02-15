Orbidet (by Rodrigo Oliveira)
----------

I developed this library as part of my master's thesis ("Orbit Determination for Low-Altitude
Satellites Using Semianalytical Satellite Theory"). It implements Orbit Determination with
Kalman Filters.

Two orbital propagation schemes were implemented: Cowell propagation[1] and Semianalytical propagation[2][3], according
to the theory developed in the aforementioned references. A simplified force model with:

      *Gravitational field harmonics (EGM96)

      *Atmospheric Drag (constant area, mass & CD) with exponential atmosphere

was coded.

The filters implemented are:

      *Gauss Initial Orbit Determination

      * Batch Least Squares

      *Extended and Unscented Cowell Kalman Filter (EKF and UKF)

      *Extended and Unscented Semianalytical Kalman Filter (ESKF and USKF)

USKF is an original contribution of my thesis.

This library was build on the Beyond Package (`github <https://github.com/galactics/beyond>`) by Jules David,
which takes care of the low-level treatment of orbital definitions (orbits, time, reference frame rotations,...).
I needed to do some changes on Beyond to better accommodate my code. These changes made in the 'orbidet.__init__.py'
file (externally to the Beyond Package).

This library is not very optimized (it's python after all :/ ). However, I did implement some modules in
Cython, which makes the code much faster. Unfortunately, I didn't maintain these modules, and they are not updated
to the latest/final/revised/clean version of orbidet. I will include them here so that you may use them, if you wish to
optimize this code.

Finally, this code is not licensed, but you are free to use it and adapt it to your projects!


Examples
----------
Some examples illustrating the use orbidet are included in the root folder of this library. Their uses are:
-observations -> This script creates measurement records of ground station passes

-propagation -> Demonstrates the setup and use of Cowell and Semianalytical propagators

-comparing_propagators -> In this file the Cowell and Semianalytical propagators are compared (in terms of accuracy)

-filter -> Main script of this package. A full simulation environment is set, and the filter algorithms
(+ initialization routines) are coded and evaluated


Package Requirements
----------

Numpy

Scipy

matplotlib

pandas

Beyond (an explicit copy of the version 0.7.2 is included here)


References
----------

[1] D. Vallado and W. McClain. Fundamentals of Astrodynamics and Applications. Space Technology
Library. Microcosm Press & Springer, 4 th edition, 2013. ISBN 978-1881883180.

[2] T. A. Ely. Mean element propagations using numerical averaging. The Journal of the Astronautical
Sciences, 61(3):275–304, 2014.

[3] T. A. Ely. Transforming mean and osculating elements using numerical methods. The Journal of
the Astronautical Sciences, 62(1):21–43, 2015.


Contacts
----------

If you have any doubts about this package, feel free to contact me through my email
rodrigo.d.oliveira@tecnico.ulisboa.pt
