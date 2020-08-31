#pragma once

struct Pos
{
	float x;
	float y;
	float z;

	Pos() {};
	Pos(float X, float Y, float Z) : x(X), y(Y), z(Z) {};
};

struct Offset
{
	float halfWidth;
	float halfHeight;
	float halfDepth;

	Offset() {};
	Offset(float hw, float hh, float hd) : halfWidth(hw), halfHeight(hh), halfDepth(hd) {};
};

class FluidObject
{
public:
	FluidObject();
	FluidObject(Pos inPos, Offset inOffset) : pos(inPos), offset(inOffset) {};
	

public:
	Pos pos = Pos(0, 0, 0);
	Offset offset = Offset(0, 0, 0);
};