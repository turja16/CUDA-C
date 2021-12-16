#include<stdio.h>
#include<stdlib.h>
#include <cuda.h>
#include <math.h>
#define DEBUG 1
#ifdef DEBUG
#define DEBUG_PRINT(fmt,args...) printf(fmt, ##args)
#else
#define DEBUG_PRINT(...)
#endif
#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE 32
#define BLOCK_QUEUE_SIZE 64
#define MAX_FRONTIER_SIZE 20

__global__ void swap_kernel( unsigned int *c_frontier_tail_d,  unsigned int *p_frontier_tail_d)
{
	//	printf("Inside swap_Kernel\n");
	if(threadIdx.x == 0)
	{
		*p_frontier_tail_d = *c_frontier_tail_d;
		*c_frontier_tail_d = 0;
		printf("Inside Swap Kernel \n\nc_frontier_tail_d = %d,p_frontier_tail_d=%d\n",*c_frontier_tail_d,*p_frontier_tail_d);
	}
	// launch a simple kernel to set *p_frontier_tail_D = *c_frontier_tail_d *c_frontier_d =0;
}
__global__ void initialize_kernel(unsigned int source, unsigned int *c_frontier_tail_d, unsigned int *p_frontier_d, unsigned int * label, unsigned int *p_frontier_tail_d,unsigned int *visited)
{
	//printf("Inside Initialize_Kernel\n");
	if(threadIdx.x == 0)
	{
		*c_frontier_tail_d = 0;
		p_frontier_d[0] = source;
		*p_frontier_tail_d = 1;
		label[source] = 0;
		visited[source] = 1;

		printf("Inside initialize kernel \nc_frontier_tail_d = %d,p_frontier_d[0]= %d,p_frontier_tail_d=%d, label[source]=%d\n",*c_frontier_tail_d,p_frontier_d[0],*p_frontier_tail_d,label[source] );
	}
}
__global__ void bfs_kernel(unsigned int* p_frontier, unsigned int *p_frontier_tail, unsigned int *c_frontier,unsigned int *c_frontier_tail, unsigned int *edges, unsigned int *dest, unsigned int*label, unsigned int *visited)
{
	//printf("Inside bfs_Kernel\n");
	__shared__ unsigned int  c_frontier_s[BLOCK_QUEUE_SIZE];
	__shared__ unsigned int  c_frontier_tail_s, our_c_frontier_tail;

	if(threadIdx.x == 0) c_frontier_tail_s = 0;
	__syncthreads();

	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadIdx.x == 0) 
	{        
		printf("edges = [");
		for (int i = 0; i < 10; i++) {
			printf("\t%d", edges[i]);
		}
		printf("]\n");
	}
	if(threadIdx.x == 0)	printf ("bfs_kernel p_frontier_tail = %d\n",*p_frontier_tail);

	if (tid < *p_frontier_tail){
		const unsigned int my_vertex = p_frontier[tid];
		printf("my_vertex = %d\n", my_vertex);
		printf("tid = %d edges[my_vertex] = %d, edges[my_vertex+1]= %d\n",tid,edges[my_vertex],edges[my_vertex+1]); 
		for(unsigned int i = edges[my_vertex]; i< edges[my_vertex+1]; ++i)
		{
			if(tid > 0)	printf("tid =%d edge = %d dest[%d] = %d\n",tid,i,i,dest[i]);
			const unsigned int was_visited = atomicExch(&(visited[dest[i]]), 1);
			if(!was_visited){
				label[dest[i]] = label[my_vertex]+1;
				const unsigned int my_tail = atomicAdd(&c_frontier_tail_s,1);
				if(my_tail < BLOCK_QUEUE_SIZE){
					c_frontier_s[my_tail] = dest[i];
					printf("c_frontier = %d\n",c_frontier_s[my_tail]);
				}else{// if full, add it to the global queue directly
					c_frontier_tail_s = BLOCK_QUEUE_SIZE;
					const unsigned int my_global_tail = atomicAdd(c_frontier_tail,1);
					c_frontier[my_global_tail] = dest[i];
				}
			}

		}
	}
	__syncthreads();


	if(threadIdx.x == 0){
		our_c_frontier_tail = atomicAdd(c_frontier_tail,c_frontier_tail_s);
		printf("c_frontier_tail = %d\n",*c_frontier_tail);
		printf("bfs_kernel label = [");
		for (int i = 0; i < 9; i++) {
			printf("\t%d", label[i]);
		}
		printf("]\n");
		printf("bfs_kernel visited = [");
		for (int i = 0; i < 9; i++) {
			printf("\t%d", visited[i]);
		}
		printf("]\n");
	}
	__syncthreads();

	for(unsigned int i = threadIdx.x; i < c_frontier_tail_s; i +=blockDim.x){
		c_frontier[our_c_frontier_tail + i] = c_frontier_s[i];
	}

	//DEBUG_PRINT("d_X[%d] = %.1f\n", i, ds_XY[threadIdx.x]);
	//DEBUG_PRINT("stride = %d: i = %d, ds_XY[%d] = %.1f\n",stride, i, threadIdx.x, ds_XY[threadIdx.x]);


}
void bfs_host(unsigned int source,unsigned int *edges, unsigned int *dest, unsigned int*label, unsigned int edges_len,unsigned int dest_len, unsigned int label_len)
{
	//allocate edges_D, dest_d, label_d and visited_d in device global memory
	//copy edges, dest and label to device global memory
	//allocate frontier_D, c_frontier_tail_d, p_frontier_tail_d in device global memory

	unsigned int * edges_d, *dest_d, *label_d, *visited_d, *frontier_d, *c_frontier_tail_d, *p_frontier_tail_d;
	unsigned int *c_frontier_d;
	unsigned int *p_frontier_d;	
	printf("Inside bfs_host\n\n");
	printf("label = [");
	for (int i = 0; i < 9; i++) {
		printf("\t%d", label[i]);
	}
	printf("]\n");
	printf("edges = [");
	for (int i = 0; i < 10; i++) {
		printf("\t%d", edges[i]);
	}
	printf("]\n");
	printf("dest = [");
	for (int i = 0; i < 14; i++) {
		printf("\t%d", dest[i]);
	}
	printf("]\n");

	cudaMalloc((void **) &edges_d, edges_len*sizeof(unsigned int));
	cudaMalloc((void **) &dest_d, dest_len*sizeof(unsigned int));
	cudaMalloc((void **) &label_d, label_len*sizeof(unsigned int));
	cudaMalloc((void **) &visited_d, label_len*sizeof(unsigned int));
	cudaMalloc((void **) &frontier_d, 2*MAX_FRONTIER_SIZE*sizeof(unsigned int));
	cudaMalloc((void **) &c_frontier_tail_d, 1*sizeof(unsigned int));
	cudaMalloc((void **) &p_frontier_tail_d, 1*sizeof(unsigned int));
	cudaMemset(frontier_d, 0, MAX_FRONTIER_SIZE* sizeof(unsigned int));	

	cudaMemset(label_d, 0, label_len* sizeof(unsigned int));	
	cudaMemset(visited_d, 0, label_len* sizeof(unsigned int));	

	cudaMemcpy(edges_d, edges, edges_len*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(dest_d, dest, dest_len*sizeof(unsigned int), cudaMemcpyHostToDevice);
	//cudaMemcpy(label_d,label, label_len*sizeof(unsigned int), cudaMemcpyHostToDevice);

	c_frontier_d = &frontier_d[0];
	p_frontier_d = &frontier_d[MAX_FRONTIER_SIZE];

	//__global__ void initialize(unsigned int source, unsigned int *c_frontier_tail_d, unsigned int *p_frontier_d, unsigned int * label, unsigned int *p_frontier_tail_d)
	initialize_kernel<<<1,1>>>(source,c_frontier_tail_d,p_frontier_d,label_d,p_frontier_tail_d, visited_d);
	cudaDeviceSynchronize();
	unsigned int p_frontier_tail = 1;
	while(p_frontier_tail > 0)	
	{
		int num_blocks = ceil(p_frontier_tail/float(BLOCK_SIZE));

		//__global__ void bfs_kernel(unsigned int* p_frontier, unsigned int *p_frontier_tail, unsigned int *c_frontier,unsigned int *c_frontier_tail, unsigned int *edges, unsigned int *dest, unsigned int*label, unsigned int *visited)
		bfs_kernel<<<num_blocks,BLOCK_SIZE>>>(p_frontier_d, p_frontier_tail_d, c_frontier_d, c_frontier_tail_d, edges_d, dest_d,label_d,visited_d);
		cudaDeviceSynchronize();
		// use cudaMemcpy to read the *c_frontier_tail value back to host and assign
		//it to p_frontier_tail for the while-loop condition test

		cudaMemcpy(label, label_d, label_len*sizeof(unsigned int), cudaMemcpyDeviceToHost);
		printf("Host label = [");
		for (int i = 0; i < 9; i++) {
			printf("\t%d", label[i]);
		}
		printf("]\n");
		cudaMemcpy(&p_frontier_tail, c_frontier_tail_d, 1*sizeof(unsigned int), cudaMemcpyDeviceToHost);
		printf("p_frontier_tail = %d\n",p_frontier_tail);


		unsigned int *temp = c_frontier_d; c_frontier_d = p_frontier_d; p_frontier_d = temp; // swap the roles

		// launch a simple kernel to set *p_frontier_tail_D = *c_frontier_tail_d *c_frontier_d =0;
		//__global__ void swap( unsigned int *c_frontier_tail_d, unsigned int *c_frontier_d,  unsigned int *p_frontier_tail_d)
		swap_kernel<<<1,1>>>(c_frontier_tail_d, p_frontier_tail_d);
		cudaDeviceSynchronize();
	}


	cudaFree(edges_d);
	cudaFree(dest_d);
	cudaFree(label_d);
	cudaFree(visited_d);
	cudaFree(frontier_d);
	cudaFree(c_frontier_tail_d);
	cudaFree(p_frontier_tail_d);
}
int main(int argc, char *argv[])
{

	unsigned int dest[15] = {1,2,3,4,5,6,7,4,8,5,8,6,8,0,6};
	unsigned int edges[10] = {0,2,4,7,9,11,12,13,15,15};
	unsigned int label[9] = {0,0,0,0,0,0,0,0,0};

	//void bfs_host(unsigned int source,unsigned int *edges, unsigned int *dest, unsigned int*label, unsigned int edges_len,unsigned int dest_len, unsigned int label_len)
	bfs_host(0, edges, dest, label,10,15,9);

	//printf("y = [");
	//for (int i = 0; i < 9; i++)
	//	printf("\t%.1f", label[i]);
	//printf("]\n");

	return 0;
}
