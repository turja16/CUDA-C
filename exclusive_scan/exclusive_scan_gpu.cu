#include<stdio.h>
#include<stdlib.h>

#define DEBUG 1
#ifdef DEBUG
#define DEBUG_PRINT(fmt,args...) printf(fmt, ##args)
#else
#define DEBUG_PRINT(...)
#endif
#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE 32
#define SECTION_SIZE 64
__global__ void Kogge_Stone_exclu_scan_kernel(float *d_X, float *d_Y, int n)
{
	extern __shared__ float ds_XY[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int section_size = blockIdx.x < blockDim.x - 1 ? blockDim.x : n % blockDim.x;
	// section_size = n for gridSize = 1
	if (i == 0)
	{
		ds_XY[threadIdx.x] = 0;
		DEBUG_PRINT("d_X[%d] = %.1f\n", i, ds_XY[threadIdx.x]);
	}
	if ( i < n) {
		ds_XY[threadIdx.x+1] = d_X[i];
		DEBUG_PRINT("d_X[%d] = %.1f\n", i+1, ds_XY[threadIdx.x+1]);
	}

	for (unsigned int stride = 1; stride < section_size; stride *= 2) {
		__syncthreads();
		if (threadIdx.x >= stride && threadIdx.x < section_size) {
			ds_XY[threadIdx.x] += ds_XY[threadIdx.x - stride];
			DEBUG_PRINT("stride = %d: i = %d, ds_XY[%d] = %.1f\n",
					stride, i, threadIdx.x, ds_XY[threadIdx.x]);
		}
	}

	if (i < n) {
		d_Y[i] = ds_XY[threadIdx.x];
	}

}
void parallel_exclu_scan(float* h_X, float* h_Y, unsigned int n)
{
	float *d_X, *d_Y;
	unsigned int size = n * sizeof(float);

	cudaMalloc((void **) &d_X, size);
	cudaMalloc((void **) &d_Y, size);
	cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);
	unsigned int gridSize = 1;
	unsigned int blockSize = n;
	unsigned int sharedMemSize = n * sizeof(float);
	Kogge_Stone_exclu_scan_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_X, d_Y, n);

	cudaMemcpy(h_Y, d_Y, size, cudaMemcpyDeviceToHost);

	cudaFree(d_X);
	cudaFree(d_Y);
}
int main(int argc, char *argv[])
{

	printf("Enter the number of elements for scan: ");
	int n;
	scanf("%d", &n);

	float *h_X;
	float *h_Y;
	h_X = (float *) malloc(n * sizeof(float));
	h_Y = (float *) malloc(n * sizeof(float));

	printf("x = [");
	for (int i = 0; i < n; i++) {
		h_X[i] = (float) i + 1;
		printf("\t%.1f", h_X[i]);
	}
	printf("]\n");

	parallel_exclu_scan(h_X, h_Y, n);

	printf("y = [");
	for (int i = 0; i < n; i++)
		printf("\t%.1f", h_Y[i]);
	printf("]\n");

	free(h_X);
	free(h_Y);
	return 0;
}
