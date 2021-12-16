#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define BLOCK_WIDTH 6


__global__ void sumRedKernel0(float* d_vec, unsigned int n)
{
	if(threadIdx.x == 0)
		printf("Sum kernel 0\n");
	extern __shared__ float ds_partialSum[];
	//Each thread loads one element from the global memory to the shared memory
	ds_partialSum[threadIdx.x] = d_vec[threadIdx.x];

	//Parallel sum reduction in the shared memory
	for (unsigned int stride =1; stride < n; stride *=2)
	{
		__syncthreads();
		if(threadIdx.x%(2*stride) ==0 && threadIdx.x + stride < n)
		{
			ds_partialSum[threadIdx.x] += ds_partialSum[threadIdx.x + stride];
		}

		//the 1st thread saves the sum from the shared memory to the global memory
		if(threadIdx.x == 0)
			d_vec[0] = ds_partialSum[0];
	}
}
__global__ void sumRedKernel1(float* d_vec, unsigned int n)
{

	if(threadIdx.x == 0)
		printf("Sum kernel 1\n");
	extern __shared__ float ds_partialSum[];

	if(threadIdx.x < n)
	{
		ds_partialSum[threadIdx.x] = d_vec[threadIdx.x];
	}
	else
	{
		ds_partialSum[threadIdx.x] = 0.0;  // padding with 0s
	}

	for (unsigned int stride = blockDim.x/2; stride >=1; stride/=2)
	{
		__syncthreads();
		if(threadIdx.x < stride)
		{
			ds_partialSum[threadIdx.x] += ds_partialSum[threadIdx.x + stride];
		}
	}
	if(threadIdx.x == 0)
		d_vec[0] = ds_partialSum[0];
}


__global__ void sumRedKernel2(float* d_vec, unsigned int n)
{
	if(threadIdx.x == 0)
		printf("Sum kernel 2\n");
	extern __shared__ float ds_partialSum[];

	//each thread loads one element from the global memry to shared memory

	if(threadIdx.x + n/2 < n)
	{
		ds_partialSum[threadIdx.x] = d_vec[threadIdx.x];
		ds_partialSum[threadIdx.x + n/2] = d_vec[threadIdx.x + n/2];
	}else
	{
		ds_partialSum[threadIdx.x] = 0.0; /// padding for odd n
	}	
	//Parrallel sum reduction in the shared memory
	for ( unsigned int stride = 1; stride < blockDim.x/2; stride *=2)
	{
		__syncthreads();

		if(threadIdx.x % (2*stride) == 0 && threadIdx.x + stride < blockDim.x/2)
		{
			ds_partialSum[threadIdx.x] += ds_partialSum[threadIdx.x+stride];
		}
		else if( (threadIdx.x -1)%(2*stride)==0 && threadIdx.x -1 + stride < blockDim.x/2)
		{
			ds_partialSum[threadIdx.x -1+blockDim.x/2] += ds_partialSum[threadIdx.x-1 + stride + blockDim.x/2];
		}
	}
	// the 1st thread saves the sum from the shared memory to the global memory
	if (threadIdx.x == 0)
	{
		d_vec[0] = ds_partialSum[0] + ds_partialSum[blockDim.x/2];
	}


}
__global__ void sumRedKernel5(float* d_vec, unsigned int n, int start, int end)
{
	if(threadIdx.x == 0)
		printf("Sum kernel 4\n");
	extern __shared__ float ds_partialSum[];
	//each thread loads one element from the global memry to shared memory
	if (threadIdx.x >= start &&  threadIdx.x <= end)
	{
		ds_partialSum[threadIdx.x-start] = d_vec[blockIdx.x *blockDim.x + threadIdx.x];
	}
	else if (threadIdx.x > end)
	{
		ds_partialSum[threadIdx.x] = 0.0;
	}else if (threadIdx.x < start)
	{
		ds_partialSum[end-threadIdx.x] = 0.0;
	}

	//printf("%d %d %.1f %.1f \n",blockIdx.x *blockDim.x + threadIdx.x,threadIdx.x,ds_partialSum[threadIdx.x],d_vec[threadIdx.x]);
	__syncthreads();
	for(unsigned int stride = blockDim.x/2; stride >=1; stride = stride >> 1)
	{
		__syncthreads();
		if(threadIdx.x < stride)
		{
			ds_partialSum[threadIdx.x] += ds_partialSum[threadIdx.x+stride];
		}
	}

	if(threadIdx.x == 0)
	{
		d_vec[blockIdx.x *blockDim.x + threadIdx.x] = ds_partialSum[0];
	}
}


__global__ void sumRedKernel4(float* d_vec, unsigned int n)
{
	if(threadIdx.x == 0)
		printf("Sum kernel 4\n");
	extern __shared__ float ds_partialSum[];
	//each thread loads one element from the global memry to shared memory
	if (blockIdx.x *blockDim.x + threadIdx.x < n)
	{
		ds_partialSum[threadIdx.x] = d_vec[blockIdx.x *blockDim.x + threadIdx.x];
	}
	else 
	{
		ds_partialSum[threadIdx.x] = 0.0;
	}
	__syncthreads();
	for(unsigned int stride = blockDim.x/2; stride >=1; stride = stride >> 1)
	{
		__syncthreads();
		if(threadIdx.x < stride)
		{
			ds_partialSum[threadIdx.x] += ds_partialSum[threadIdx.x+stride];
		}
	}

	if(threadIdx.x == 0)
	{
		d_vec[blockIdx.x *blockDim.x + threadIdx.x] = ds_partialSum[0];
	}
}


float sumRed(float* h_vec, unsigned int n, unsigned int version)
{
	unsigned int size = n * sizeof(float);
	float *d_vec;
	cudaMalloc((void **) &d_vec, size);
	cudaMemcpy(d_vec, h_vec, size, cudaMemcpyHostToDevice);
	unsigned int blockSize, gridSize, sharedMemSize; 

	switch (version) {
		case 0:
			gridSize = 1;
			blockSize = n;
			sharedMemSize = n * sizeof(float);
			sumRedKernel0<<<gridSize, blockSize, sharedMemSize>>>(d_vec, n); break;
		case 1:
			gridSize = 1;
			blockSize = 1 << ((int) ceil(log2(n)));
			sharedMemSize = blockSize * sizeof(float); 
			sumRedKernel1<<<gridSize, blockSize, sharedMemSize>>>(d_vec, n); break;
		case 2: 
			gridSize = 1;
			blockSize = 1 << ((int) ceil(n/2));
			sharedMemSize = blockSize * sizeof(float); 
			sumRedKernel2<<<gridSize, blockSize, sharedMemSize>>>(d_vec, n); break;	
			// Launch the kernel sumRedKernel2 with execution configuration parameters. break;
		case 5:
			int start, end;
			gridSize = 1;
			printf("start position:");
			scanf("%d",&start);
			printf("end position:");
			scanf("%d", &end);
			blockSize = 1 << ((int) ceil(log2(end-start+1)));
			printf("blockSize = %d", blockSize);
			sharedMemSize = blockSize * sizeof(float);
			sumRedKernel5<<<gridSize, blockSize, sharedMemSize>>>(d_vec, n,start,end);
			break;
			//	case 4:
			//		gridSize = 4;
			//                blockSize = 1 << ((int) ceil(log2(n)));
			//		blockSize = blockSize/gridSize;
			//                sharedMemSize = blockSize * sizeof(float);
			//                sumRedKernel4<<<gridSize, blockSize, sharedMemSize>>>(d_vec, n); break;
	}
	float h_sum = 0.0;
	cudaMemcpy(&h_sum, d_vec, sizeof(float), cudaMemcpyDeviceToHost); cudaFree(d_vec);
	return h_sum;
}
int main(int argc, char *argv[])
{
	int version = 0;
	if (argc > 1) {
		version = atoi(argv[1]);
		printf("Kernel version = %d\n", version); }
	printf("Enter the number of elements to be summed up: "); unsigned int n;
	scanf("%d", &n);
	float *h_vec = (float *) malloc(n * sizeof(float));
	for (int i = 0; i < n; i++)
	{
		h_vec[i] = (float) i;
	}

	float sum = sumRed(h_vec, n, version); printf("sum = %.1f\n", sum); free(h_vec);
	return 0;
}

