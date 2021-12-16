#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define WIDTH 8
#define HEIGHT 8
#define O_TILE_SIZE 4

#define TILE_WIDTH 4
#define MASK_HEIGHT 3
#define MASK_WIDTH 3
#define w_x (O_TILE_SIZE+MASK_WIDTH -1)
#define w_y (O_TILE_SIZE+MASK_HEIGHT -1)
#define MASK_RADIUS_X MASK_WIDTH/2
#define MASK_RADIUS_Y MASK_HEIGHT/2


__constant__ float const_mem[MASK_HEIGHT*MASK_WIDTH];


__global__ void convolution_2D_basic_kernel(float* d_O, float* d_I, int height, int width, int pitch, int Mask_Height,int Mask_Width)
{
	__shared__ float I_ds[w_y][w_x];

	// first batch loading
	int tx = threadIdx.x, ty = threadIdx.y;
	int dest = threadIdx.y*O_TILE_SIZE + threadIdx.x;
	int destY = dest / w_x;
	int destX = dest % w_x;

	int srcY = blockIdx.y*O_TILE_SIZE+ destY - MASK_RADIUS_X;    	
	int srcX = blockIdx.x*O_TILE_SIZE+ destX - MASK_RADIUS_Y;

	int src = srcX + srcY*pitch; //width? pitch?

	//	int src = srcX + srcY*width; //width? pitch?

	if(srcY >=0 && srcY < height && srcX >=0 && srcX < width) 
	{

		I_ds[destY][destX] = d_I[src];

		//printf ("%f %f \n",d_I[threadIdx.y*width + threadIdx.x], I_ds[destY][destX]);
	} else
	{

		I_ds[destY][destX] = 0.0f;
	}
	__syncthreads();
	///
	if (tx == 0 && ty == 0)
	{
		printf("\n");
		for(int i = 0; i < w_x; i++)
		{
			for(int j = 0; j< w_y; j++)
			{
				//			printf ("%0.1f \t", I_ds[i][j]);
			}
			//		printf("\n");
		}
		for( int i = 0; i < MASK_HEIGHT; i++)
		{       
			for(int j = 0; j< MASK_WIDTH; j++)
			{       
				//               output += const_mem[i* MASK_HEIGHT+j] * I_ds[i+ty][j+tx];
				//printf ("%0.1f \t", const_mem[i*MASK_HEIGHT+j]);
				//         		printf ("%0.1f \t", I_ds[i+ty][j+tx]);
			}
			//		printf("\n");
		}

	}
	////


	// 2nd batch loading
	for (int iter =1; iter <= (w_x*w_y)/(O_TILE_SIZE*O_TILE_SIZE);iter++)
	{
		dest = threadIdx.y*O_TILE_SIZE + threadIdx.x + iter*O_TILE_SIZE*O_TILE_SIZE;
		destY = dest / w_x;
		destX = dest % w_x;
		srcY = blockIdx.y*O_TILE_SIZE+ destY - MASK_RADIUS_Y;    	
		srcX = blockIdx.x*O_TILE_SIZE+ destX - MASK_RADIUS_X;
		src = srcX + srcY*pitch; //width? pitch?
		//	src = srcX + srcY*width; //width? pitch?
		if (destY < w_y && destX < w_x)
		{
			if(srcY >=0 && srcY < height && srcX >=0 && srcX < width) 
			{
				I_ds[destY][destX] = d_I[src];
			} else
			{
				I_ds[destY][destX] = 0.0f;
			}
		}
	}


	__syncthreads();
	float output = 0.0f;
	//printf("tx = %d, ty = %d\n",tx,ty);
	if(ty < O_TILE_SIZE && tx < O_TILE_SIZE)
	{
		int i,j;
		for( i = 0; i < MASK_HEIGHT; i++)
		{
			for( j = 0; j< MASK_WIDTH; j++)
			{

				output += const_mem[i* MASK_HEIGHT+j] * I_ds[i+ty][j+tx];
			}
		}
		//	printf("output = %.1f const mem %.1f input %.1f \n",output,const_mem[i* MASK_HEIGHT+j], I_ds[i+ty][j+tx]);
		//	printf("output = %.1f \n",output);
		i = blockIdx.y * O_TILE_SIZE + threadIdx.y;
		j = blockIdx.x * O_TILE_SIZE + threadIdx.x;
		if(i < height && j < width){
			d_O[i*width+j] = output;
		}
		__syncthreads();
	}
}
void convolution_2D(float* h_O, float* h_M, float*h_I, int height,int width, int Mask_Height,int Mask_Width)
{
	int mask_size = Mask_Height*Mask_Width*sizeof(float);
	int pitch = 2*((int)ceil(width/2.0));
	int input_size = height*pitch*sizeof(float);
	//int input_size = height*width*sizeof(float);
	int output_size = height*width*sizeof(float);

	float* d_I,*d_O;
	cudaMalloc((void**)&d_I, input_size);	
	cudaMalloc((void**)&d_O, output_size);


	// copy the mask matrix M from host memory to constant memory of GPU
	cudaMemcpyToSymbol(const_mem,h_M,mask_size);
	// copy the input matrix N from host memory to global memory of GPU
	cudaMemcpy(d_I, h_I, input_size, cudaMemcpyHostToDevice);

	//Initialize execution configuration dimBlock and dimGrid with type dim3

	dim3 dimBlock(O_TILE_SIZE,O_TILE_SIZE);
	dim3 dimGrid(((width-1)/O_TILE_SIZE)+1,((height-1)/O_TILE_SIZE)+1);
	//launch the kernel convolution_2d_basic_kernel
	convolution_2D_basic_kernel<<<dimGrid, dimBlock>>>(d_O, d_I, height, width,pitch, Mask_Height, Mask_Width);	
	//convolution<<<dimGrid, dimBlock>>>(d_O, d_I,0, width,height);	

	cudaMemcpy(h_O , d_O, output_size, cudaMemcpyDeviceToHost);

	cudaFree(d_I);
	cudaFree(d_O);

}



int main(int argc,char*argv[]){
	int mask_size = MASK_HEIGHT * MASK_WIDTH * sizeof(float);
	int pitch = 2*((int)ceil(WIDTH/2.0));
	int input_size = HEIGHT * pitch * sizeof(float);
	//	int input_size = HEIGHT * WIDTH * sizeof(float);
	int output_size = HEIGHT * WIDTH * sizeof(float);

	float* h_M = (float*)malloc(mask_size);
	float* h_I = (float*)malloc(input_size);
	float* h_O = (float*)malloc(output_size);

	printf("Mask M:\n");
	for(int i = 0; i <MASK_HEIGHT; i++){
		for(int j=0; j < MASK_WIDTH; j++)
		{
			h_M[i*MASK_WIDTH+j] = 1;//rand()%10 + 1;
			printf("%.1f\t",h_M[i*MASK_WIDTH+j]);
		}
		printf("\n");
	}

	printf("Input Image N :\n");

	for(int i = 0; i <HEIGHT; i++){
		for(int j=0; j < WIDTH; j++)
		{
			h_I[i*pitch+j] = 1;//rand()%10 + 1;
			//                      h_I[i*WIDTH+j] = 1;//rand()%10 + 1;
			//printf("%.1f\t",h_I[i*pitch+j]);
			printf("%.1f\t",h_I[i*WIDTH+j]);
		}
		printf("\n");
	}


	convolution_2D(h_O, h_M, h_I, HEIGHT, WIDTH,MASK_HEIGHT,MASK_WIDTH);

	printf("Output IMAGE P:\n");        
	for(int i = 0; i <HEIGHT; i++)
	{
		for(int j=0; j < WIDTH; j++)
		{
			printf("%.1f\t",h_O[i*WIDTH+j]);
		}
		printf("\n");
	}

	free(h_M);
	free(h_I);
	free(h_O);
	return 0;

}
