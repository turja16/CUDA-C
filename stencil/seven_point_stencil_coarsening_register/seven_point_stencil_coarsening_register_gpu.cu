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



__global__ void stencil_kernel(float* in, float* out, unsigned int N) { 
	int iStart = blockIdx.z*OUT_TILE_DIM;
	int j = blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1; 
	int k = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1; 
	float inPrev; 
	__shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
	//__shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM]; 

	float inCurr; 
	float inNext;
	if(iStart-1 >= 0 && iStart-1 < N && j >= 0 && j < N && k >= 0 && k < N) { 
		inPrev = in[(iStart - 1)*N*N + j*N + k]; 
	} 
	if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) { 
		inCurr = in[iStart*N*N + j*N + k]; 
		inCurr_s[threadIdx.y][threadIdx.x] = inCurr; 
	} 
	for(int i = iStart; i < iStart + OUT_TILE_DIM; ++i) { 
		if(i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) { 
			inNext = in[(i + 1)*N*N + j*N + k]; 
			//	inNext_s[threadIdx.y][threadIdx.x] = inNext; 
		} 
		__syncthreads(); 
		if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) { 
			if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) { 
				out[i*N*N + j*N + k] = c0*inCurr 
					+ c1*inCurr_s[threadIdx.y][threadIdx.x-1] 
					+ c2*inCurr_s[threadIdx.y][threadIdx.x+1] 
					+ c3*inCurr_s[threadIdx.y+1][threadIdx.x] 
					+ c4*inCurr_s[threadIdx.y-1][threadIdx.x] 
					+ c5*inPrev 
					+ c6*inNext; 
			} 
		} 
		__syncthreads(); 
		inPrev = inCurr; 
		inCurr = inNext; 
		inCurr_s[threadIdx.y][threadIdx.x] = inNext;//inNext_s[threadIdx.y][threadIdx.x]; 
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

	dim3 dimBlock(IN_TILE_DIM,IN_TILE_DIM);
	dim3 dimGrid(N/OUT_TILE_DIM,N/OUT_TILE_DIM);
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
