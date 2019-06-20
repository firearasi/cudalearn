#include <iostream>
#define N 640000
#define TPB 32

__host__
float scale(int i, int n)
{
	return float(i) / ((float)(n-1));
}
__device__
float distance(float x1, float x2)
{
  return sqrt((x2 - x1)*(x2 - x1));
}


__global__ 
void distanceKernel(float *d_out, float *d_in, float ref)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    float x=d_in[i];
    d_out[i]=distance(x,ref);
    //printf("blockIdx = %2d, threadId = %2d, i = %2d: dist from %f to %f is %f.\n",
    //    blockIdx.x, threadIdx.x, i, x,ref,d_out[i]);
}


int main()
{
	float *d_in = 0;
	float *d_out = 0;
	const float ref = 0.5;
	cudaMallocManaged(&d_in, N*sizeof(float));
	cudaMallocManaged(&d_out, N*sizeof(float));

	for(int i=0; i < N;i++)
		d_in[i]=scale(i,N);
		
	distanceKernel<<<N/TPB, TPB>>>(d_out, d_in, ref);
	cudaDeviceSynchronize();
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}



