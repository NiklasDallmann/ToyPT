struct CoordinateBufferPointer
{
	float *x;
	float *y;
	float *z;
};

struct Mesh
{
	uint triangleOffset;
	uint triangleCount;
	uint materialOffset;
};

struct IntersectionInfo
{
	struct Mesh *mesh;
	uint triangleOffset;
	float u;
	float v;
};
