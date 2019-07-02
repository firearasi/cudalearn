#include <stdio.h>
#define N 1000
#define TPB 32 // Threads per block



__global__ void  summationKernel(int *d_array, int n, int *d_res)
{
	const int idx=threadIdx.x+blockIdx.x*blockDim.x;
	const int s_idx=threadIdx.x;
	__shared__ int s_array[TPB];
	if(idx<n)
		s_array[s_idx]=d_array[idx];
	else
	{
		s_array[s_idx]=0;
		return;
	}
	__syncthreads();

	for(int s=blockDim.x/2;s>0;s>>=1)
	{
		if(s_idx<s)
		{
			s_array[s_idx]+=s_array[s_idx+s];
		}
	}
	__syncthreads();

	if(s_idx==0)
	{
		atomicAdd(d_res, s_array[0]);
	}
}

void summationLauncher(int *array, int n)
{
	int *d_array=0;
	cudaMalloc(&d_array,n*sizeof(int));
	cudaMemcpy(d_array,array,n*sizeof(int),cudaMemcpyHostToDevice);
	int *d_res;
	cudaMalloc(&d_res,sizeof(int));
	cudaMemset(d_res,0,sizeof(int));
	int blocks = (N+TPB-1)/TPB;

	summationKernel<<<blocks,TPB>>>(d_array,n,d_res);
	int res;
	cudaMemcpy(&res,d_res,sizeof(int), cudaMemcpyDeviceToHost);
	printf("Sum is %d\n", res);
	cudaFree(d_array);
	cudaFree(d_res);
}




int main()
{
	int array[N];
	for(int i=0;i<N;i++)
	{
		array[i]=i;
	}
	summationLauncher(array,N);
	return 0;
}
