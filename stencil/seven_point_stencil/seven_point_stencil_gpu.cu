#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define DEBUG 1
#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) printf(fmt, ## args)
#else
#define DEBUG_PRINT(...) 
#endif

#define M 6
#define c0 1
#define c1 1
#define c2 1
#define c3 1
#define c4 1
#define c5 1
#define c6 1
#define STENCIL_DIM 3
#define STENCIL_POINT 7
#define OUT_TILE_DIM 6
#define IN_TILE_DIM 8 // (OUT_TILE_DIM + ((STENCIL_POINT-1)/STENCIL_DIM))


__global__ void stencil_kernel(float* in, float* out, unsigned int N) 
{ 
	int i = blockIdx.z*OUT_TILE_DIM + threadIdx.z - 1;
	int j = blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1;
	int k = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1;
	__shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM]; 
	//int N = M;
	if(i>=0&&i<N&&j>=0&&j<N&&k>=0&&k<N)
	{
		in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N + k];
		//	DEBUG_PRINT("Block %d, %d, %d:  = %.1f\n", blockIdx.z, blockIdx.y, blockIdx.x, 
		//	 in_s[threadIdx.z][threadIdx.y][threadIdx.x]);
	}
	__syncthreads();

	if(i>=1&&i<N-1&&j>=1&&j<N-1&&k>=1&&k<N-1)
	{ 
		if(threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM-1 && threadIdx.y >= 1&& threadIdx.y<IN_TILE_DIM-1 && threadIdx.x>=1 && threadIdx.x<IN_TILE_DIM-1)
		{ 
			out[i*N*N + j*N + k] = c0*in_s[threadIdx.z][threadIdx.y][threadIdx.x]
				+ c1*in_s[threadIdx.z][threadIdx.y][threadIdx.x-1] 
				+ c2*in_s[threadIdx.z][threadIdx.y][threadIdx.x+1] 
				+ c3*in_s[threadIdx.z][threadIdx.y-1][threadIdx.x] 
				+ c4*in_s[threadIdx.z][threadIdx.y+1][threadIdx.x] 
				+ c5*in_s[threadIdx.z-1][threadIdx.y][threadIdx.x] 
				+ c6*in_s[threadIdx.z+1][threadIdx.y][threadIdx.x];
		}
	}
}

void stencil(float* h_A, float* h_O,unsigned int N)
{
	float* d_A,*d_O;

	int input_size = N * N * N * sizeof(float);
	int output_size = N * N * N * sizeof(float);
	cudaMalloc((void**)&d_A, input_size);
	cudaMalloc((void**)&d_O, output_size);
	cudaMemcpy(d_A, h_A, input_size, cudaMemcpyHostToDevice);

	dim3 dimBlock(IN_TILE_DIM,IN_TILE_DIM,IN_TILE_DIM);
	dim3 dimGrid(N/OUT_TILE_DIM,N/OUT_TILE_DIM, N/OUT_TILE_DIM);
	stencil_kernel<<<dimGrid, dimBlock>>>(d_A, d_O, N);	

	cudaMemcpy(h_O , d_O, output_size, cudaMemcpyDeviceToHost);


	cudaFree(d_A);
	cudaFree(d_O);

}



int main()
{
	int N = M;
	size_t size = N*N*N* sizeof(float);
	float* h_A = (float*)malloc(size);
	float* h_O = (float*)malloc(size);
	printf("Input Matrix A:\n");
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			for(int k = 0; k < N; k++) {
				h_A[i * N * N + j * N + k] = rand() % 1 + 1;
				printf("%.1f\t", h_A[i * N * N + j * N + k]);
			}
			printf("\n");
		}
		printf("\n");
	}


	stencil(h_A, h_O,N);

	printf("\nOutput image O =\n");
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			for(int k = 0; k < N; k++) {
				printf("%.1f\t", h_O[i * N * N + j * N + k]);
			}
			printf("\n");
		}
		printf("\n");
	}	
	free(h_A);
	free(h_O);
	return 0;

}
