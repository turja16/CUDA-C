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

#define TILE_DIM 16

#define BLOCK_ROWS  16

__global__ void transposeF(float* odata, float* idata, int width, int height)
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];

	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;

	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;


	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
	{
		tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
	}

	__syncthreads();

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
	{
		odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
	}

}
__global__ void transpose(int* odata, int* idata, int width, int height)
{
	__shared__ int tile[TILE_DIM][TILE_DIM + 1];

	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;

	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;


	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
	{
		tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
	}

	__syncthreads();

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
	{
		odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
	}

}

__global__ void padKernel(float *d_M, float *Values, int *Columns, int size, int width)
{
	int rowIndex = threadIdx.x + blockDim.x* blockIdx.x;
	int rowStart = rowIndex * size;
	int rowStartELL = rowIndex * width;
	int count = 0;
	for (int n = 0; n < size; n++)
	{
		float el = d_M[rowStart + n];

		if (el != 0)
		{
			Values[rowStartELL + count] = el;

			Columns[rowStartELL + count] = n;
			count++;
		}

	}
}


int nonZeroCounter(float * data, int size)
{
	int count = 0;

	for(int i = 0; i < size; i++) 
	{
		for(int j = 0; j < size; j++) 
		{
			if(data[i*size+j] != 0)
			{
				count++;
			}
		}
	}
	return count;

}
void print_vec(float * h_M, int n)
{
	for(int i = 0; i < n; i++)
	{
		printf("%.0f ",h_M[i]);
	}
	printf("\n");
}
void print_int(int *h_M, int n)
{
	for(int i =0; i< n; i++)
	{
		printf("%d ",h_M[i]);
	}
	printf("\n");
}
void print_fl(float * h_M, int height, int width)
{
	printf("Print data ::\n");
	for (int i = 0; i < height; i++)
	{
		for(int j = 0; j < width;j++)
		{

			printf(" %.0f", h_M[i*width+j]);
		}
		printf("\n");
	}
}

void print_cl(int * h_M, int height, int width)
{
	printf("Print index ::\n");

	for (int i = 0; i < height; i++)
	{
		for(int j =0; j < width;j++)
		{			
			printf(" %d", h_M[i*width+j]);
		}
		printf("\n");
	}
}
void convert_to_csr(float * h_M, float * h_data, int *h_col_index, int * h_row_ptr, int size)
{
	int data_index =0, count = 0;        
	h_row_ptr[0] = 0;
	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			if(h_M[i*size+j] != 0)
			{
				h_data[data_index] = h_M[i*size + j];
				h_col_index[data_index++] = j;
			}
		}
		h_row_ptr[i+1] = data_index;
	}
	printf("Data ::");
	print_vec(h_data,data_index);
	printf("Col Index ::");
	print_int(h_col_index,data_index);
	printf("Row pointer ::");
	print_int(h_row_ptr,size+1);
}
__global__ void SpMV_CSR(int num_rows, float *data, int *col_index, int *row_ptr, float *x, float *y) 
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < num_rows) 
	{
		float dot = 0;
		int row_start = row_ptr[row];
		int row_end =   row_ptr[row+1];

		for (int elem = row_start; elem < row_end; elem++) { 
			dot += data[elem] * x[col_index[elem]];
		}
		y[row] += dot;
	}

}

__global__ void SpMV_ELL(int num_rows, float *data, int *col_index, int num_elem, float *x, float *y) 
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;


	if (row < num_rows) 
	{
		float dot = 0;
		for (int i = 0; i < num_elem; i++) 
		{ 
			if(col_index[row+i*num_rows] != -1)
			{
				dot += data[row+i*num_rows] * x[col_index[row+i*num_rows]];
			}
		}
		y[row] += dot;

	}

}








__global__ void countNonZeroPerRowKernel( float* d_vec, int *d_nz_counter,int n)


{

	extern __shared__ float ds_partialSum[];

	ds_partialSum[threadIdx.x] = d_vec[ blockIdx.x *blockDim.x +  threadIdx.x];

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
			d_nz_counter[blockIdx.x] = ds_partialSum[0];
	}

}





int max_row_size (int *data, int size)
{
	int max=0;
	for(int i = 0; i< size;i++)
	{
		if(data[i]> max)
		{
			max = data[i];
		}
	}
	return max;
}
void ell_host(float* h_M, float* h_Y,float *h_X, int size)
{

	float* d_M,*d_Values,*d_Values_t, *d_X, *d_Y;
	int *d_Columns, *d_Columns_t;
	cudaMalloc((void**)&d_M, size*size * sizeof(float));
	int *d_nz_counter,*d_col_index;
	cudaMemcpy(d_M, h_M, sizeof(float) * size* size, cudaMemcpyHostToDevice);
	int* h_nz_counter = (int*)malloc(size*sizeof(int));
	cudaMalloc((void**)&d_nz_counter, size* sizeof(int));

	cudaMemcpy(d_nz_counter, h_nz_counter, sizeof(float) * size, cudaMemcpyHostToDevice);
	int gridSize = size;
	int blockSize = size;//1 << ((int) ceil(log2(size)));
	int sharedMemSize = size * sizeof(float);
	countNonZeroPerRowKernel<<<gridSize,blockSize,sharedMemSize>>>(d_M,d_nz_counter,size);

	cudaMemcpy(h_nz_counter, d_nz_counter, sizeof(float)*size, cudaMemcpyDeviceToHost);
	print_int(h_nz_counter,size);
	int rowMax = max_row_size(h_nz_counter, size);
	printf("Row Max : %d\n",rowMax);
	int width = rowMax, height = size;

	float* h_Values = (float*)malloc(width*height*sizeof(float));

	int* h_Columns = (int*)malloc((width*height)*sizeof(int));

	memset(h_Values, -1, height*width*sizeof(float));
	memset(h_Columns, -1, height*width*sizeof(int));
	cudaMalloc((void**)&d_M, size*size* sizeof(float));
	cudaMalloc((void**)&d_Values, width*height* sizeof(float));
	cudaMalloc((void**)&d_Columns, (width*height)* sizeof(int));

	cudaMemcpy(d_M, h_M, sizeof(float) * size* size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Values, h_Values, sizeof(float) * width* height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Columns, h_Columns, sizeof(int) * width* height, cudaMemcpyHostToDevice);

	padKernel<<<1,size>>>(d_M,d_Values,d_Columns, height,width);

	cudaMemcpy(h_Values , d_Values, sizeof(float)*height*width, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Columns , d_Columns, sizeof(int)*height*width, cudaMemcpyDeviceToHost);

	print_cl(h_Columns,height,width);


	cudaMalloc((void**)&d_Columns_t, (width*height)* sizeof(int));

	cudaMemcpy(d_Columns, h_Columns, sizeof(int) * width* height, cudaMemcpyHostToDevice);


	dim3 dimBlock(height+1, width+1, 1);
	dim3 dimGrid(1,1, 1);


	transpose<<<dimGrid, dimBlock>>>(d_Columns_t,d_Columns, width, height);


	cudaMemcpy(h_Columns , d_Columns_t, sizeof(int)*height*width, cudaMemcpyDeviceToHost);
	print_cl(h_Columns,width,height);

	cudaMalloc((void**)&d_Values_t, (width*height)* sizeof(float));

	cudaMemcpy(d_Values, h_Values, sizeof(int) * width* height, cudaMemcpyHostToDevice);


	dim3 dimBlockF(height+1, width+1, 1);
	dim3 dimGridF(1,1, 1);

	transposeF<<<dimGridF, dimBlockF>>>(d_Values_t,d_Values, width, height);


	cudaMemcpy(h_Values , d_Values_t, sizeof(int)*height*width, cudaMemcpyDeviceToHost);

	cudaMalloc((void**)&d_Values, width*height* sizeof(float));
	cudaMalloc((void**)&d_Columns, (width*height)* sizeof(int));

	cudaMemcpy(d_Values, h_Values, sizeof(float) * width* height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Columns, h_Columns, sizeof(int) * width* height, cudaMemcpyHostToDevice);



	cudaMalloc((void**)&d_X, size* sizeof(float));
	cudaMalloc((void**)&d_Y, size* sizeof(float));
	cudaMemcpy(d_X, h_X, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, h_Y, sizeof(float)*size, cudaMemcpyHostToDevice);

	SpMV_ELL<<<1,size>>>(size,d_Values,d_Columns,width,d_X,d_Y);
	cudaMemcpy(h_Y , d_Y, sizeof(float)*size, cudaMemcpyDeviceToHost);
	cudaFree(d_M);
	cudaFree(d_Y);
	cudaFree(d_X);

}
void csr_host(float* h_M, float* h_Y,float *h_X, int size)
{
	float* d_M,*d_X,*d_Y;
	int *d_row_ptr,*d_col_index;
	int nonZeros = nonZeroCounter(h_M, size);
	printf("Non Zero Element: %d\n",nonZeros);
	float* h_data = (float*)malloc(nonZeros*sizeof(float));
	int* h_col_index = (int*)malloc(nonZeros*sizeof(int));
	int* h_row_ptr = (int*)malloc((size+1)*sizeof(int));
	convert_to_csr(h_M, h_data, h_col_index, h_row_ptr, size);

	cudaMalloc((void**)&d_M, size*size * sizeof(float));
	cudaMalloc((void**)&d_X, size* sizeof(float));
	cudaMalloc((void**)&d_Y, size* sizeof(float));
	cudaMalloc((void**)&d_col_index, nonZeros* sizeof(int));
	cudaMalloc((void**)&d_row_ptr, (size+1)* sizeof(int));

	cudaMemcpy(d_M, h_data, sizeof(float) *nonZeros, cudaMemcpyHostToDevice);
	cudaMemcpy(d_X, h_X, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, h_Y, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_col_index, h_col_index, sizeof(int)*nonZeros, cudaMemcpyHostToDevice);
	cudaMemcpy(d_row_ptr, h_row_ptr, sizeof(int)*(size+1), cudaMemcpyHostToDevice);


	//int threadsPerBlock = size;
	//int blocksPerGrid = (( threadsPerBlock+threadsPerBlock-1)/threadsPerBlock );
	print_vec(h_X,size);
	SpMV_CSR<<<1,size>>>(size,d_M,d_col_index,d_row_ptr,d_X,d_Y); 

	cudaMemcpy(h_Y , d_Y, sizeof(float)*size, cudaMemcpyDeviceToHost);


	cudaFree(d_M);
	cudaFree(d_Y);
	cudaFree(d_X);
	cudaFree(d_col_index);
	cudaFree(d_row_ptr);

}



int main()
{
	int size;
	printf("Enter size of the matrix N");
	scanf("%d",&size);

	float* h_M = (float*)malloc(size*size*sizeof(float));
	float* h_X = (float*)malloc(size*sizeof(float));
	float* h_Y = (float*)malloc(size*sizeof(float));
	int * nonZerosCount = (int*)malloc(size*sizeof(int));
	int * JDS_rowIndex = (int*)malloc(size*sizeof(int));

	for(int i = 0; i < size; i++) JDS_rowIndex[i] = i;
	for(int i = 0; i < size; i++) h_X[i] = 1;
	for(int i = 0; i < size; i++) h_Y[i] = 0;

	memset(nonZerosCount, 0, size*sizeof(int));
	memset(h_M, 0, size*size*sizeof(float));


	printf("Input Matrix :\n");

	for(int i = 0; i < size; i++) {
		for(int j = 0; j < size; j++) {
			h_M[i * size  + j ] = rand() % 2;
			printf("%.0f\t", h_M[i * size  + j ]);
		}
		printf("\n");
	}

	printf ("\n\nCSR Kernel Running....\n\n");
	csr_host(h_M, h_Y, h_X, size);

	printf("Y ::");

	print_vec(h_Y,size);

	printf ("\n\nELL Kernel Running....\n\n");
	ell_host(h_M, h_Y, h_X, size);
	printf("Y ::");
	print_vec(h_Y,size);
	free(h_M);
	free(h_X);
	free(h_Y);
	free(nonZerosCount);
	free(JDS_rowIndex);
	return 0;

}
