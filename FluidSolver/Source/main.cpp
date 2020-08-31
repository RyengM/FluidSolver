#include "FluidSolver.h"

int main()
{
	FluidSolver scene;

	FluidObject fluid(Pos(0, 0, 0), Offset(1.2, 1.2, 2.56));
	scene.AddObject(fluid);

	scene.Render();
	
	return 0;
}