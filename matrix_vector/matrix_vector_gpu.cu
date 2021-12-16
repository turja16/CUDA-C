#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define NUM_COL 2
#define NUM_ROW 3

struct Dim2{
	unsigned char nc;
	unsigned char nr;
};



__global__ void  matVecMulKernel(float* d_A, float* d_B, float* d_C, struct Dim2 dim)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;


	float sum = 0;
	if (tid < dim.nr)
	{
		for(int i =0; i< dim.nc; i++)
		{

			sum += d_C[i]*d_B[(tid*dim.nc)+i];
		}
		d_A[tid] = sum;
	}
}
void matVecMul(float* h_A, float* h_B, float* h_C, struct Dim2 dim)
{

	float* d_A;
	cudaMalloc((void**)&d_A,dim.nr* sizeof(float));

	float* d_B;
	cudaMalloc((void**)&d_B,dim.nr*dim.nc* sizeof(float));
	float * d_C;
	cudaMalloc((void**)&d_C,dim.nc* sizeof(float));
	cudaMemcpy(d_B, h_B, dim.nr*dim.nc* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C,dim.nc* sizeof(float) , cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocksPerGrid = ((threadsPerBlock +dim.nr-1)/threadsPerBlock );
	matVecMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C,dim);	

	cudaMemcpy(h_A , d_A,dim.nr* sizeof(float) , cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}



int main()
{
	int x;
	float* h_A = (float*)malloc(NUM_ROW *sizeof(float));
	float* h_B =  (float*)malloc(NUM_ROW*NUM_COL*sizeof(float));	
	float* h_C  =  (float*)malloc(NUM_COL*sizeof(float));

	//scanf("%d",&x);
	//scanf("%d",&x);
	printf("B= \n");

	for(int row = 0; row <NUM_ROW; row++)
	{
		for(int col = 0; col  <NUM_COL; col++){
			int offset = row *NUM_COL + col;

			h_B[offset] = (float)(rand()%10);
			printf("%.1f\t",h_B[offset]);
		}
		printf("\n");

	}
	//scanf("%d",&x);
	printf("\nC = \n");
	for(int col =0; col< NUM_COL;col++)
	{
		h_C[col] = (float) (rand()%10);
		printf("%.1f\n",h_C[col]);
	}

	//scanf("%d",&x);
	struct Dim2 dim;
	dim.nc = NUM_COL;
	dim.nr = NUM_ROW;
	matVecMul(h_A,h_B,h_C,dim);


	printf("\nC = \n");
	for(int row = 0; row <NUM_ROW; row++)
	{	
		printf("%.1f\n",h_A[row]);
	}
	cudaFree(h_A);
	cudaFree(h_B);
	cudaFree(h_C);
	return 0;

}
