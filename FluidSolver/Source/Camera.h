#pragma once

struct Resolution
{
	int x;
	int y;

	Resolution() {};
	Resolution(int X, int Y) : x(X), y(Y) {};
};

class Camera
{
public:
	Camera(int resolutionX, int resolutionY);

	inline Resolution GetResolution()
	{
		return resolution;
	}

private:
	Resolution resolution;
};