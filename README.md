# FEM-Helmholtz Problem

![One numerical solution](https://github.com/FredericWantiez/FEM-Helmholtz/blob/master/Images/solution6.png)

Implementation of a Finite Element Method to solve a simplified Helmholtz problem. The method uses the Galerkin approach on the subset of P1-Lagrange functions.
The mesh was generated using FreeFM++ by M. Claeys from university Paris VI (https://www.ljll.math.upmc.fr/~claeys/4M054.html)
The program computes both a solution of the given problem and its given vibrating modes. 

![Vibrating modes](https://github.com/FredericWantiez/FEM-Helmholtz/blob/master/Images/modes.png "Vibrating modes")

To compute solutions in a reasonable amount of time, the scripts uses sparse representation of matrices. You need the Scipy and Maplotlib libraries in order to make the script work.

To launch the script, the command is as follow:
`python main.py -mesh_index -omega --save_fig --alpha`

For example, the following command:
`python main.py 6 4 --save_fig=False --alpha=90`
will plot the solution and the vibrating modes without saving them. The mesh used is the number 6 with a pulse of 4 radians and an incident angle of 90°. 
