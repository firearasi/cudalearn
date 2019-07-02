#define W 500
#define H 500
#define D 500
#define TX 32
#define TY 32
#define TZ 32

int divUp(int a, int b){return (a+b-1)/b;}
__device__ float distance(int c,int r, int s ,float3 pos)
{
	return sqrtf((c-pos.x)*(c-pos.x)+(r-pos.y)*(r-pos.y)+(s-pos.z)*(s-pos.z));

}

__global__
void distanceKernel(float *d_out, int w, int h, int d,float3 pos)
{
	const int c = blockIdx.x*blockDim.x+threadIdx.x;
	const int r = blockIdx.y*blockDim.y+threadIdx.y;
	const int s = blockIdx.z*blockDim.z+threadIdx.z;
	const int i = c+r*w+s*w*h;
	if((c>=w) || (r>=h) || (s>=d)) return;

	d_out[i]=distance(c,r,s,pos);
}

int main()
{
	float *out=(float *)calloc(W*H*D, sizeof(float));
	float *d_out;
	cudaMalloc(&d_out, W*H*D*sizeof(float));
	const float3 pos={0.0f,0.0f,0.0f};
	const dim3 blockSize(TX,TY,TZ);
	const int bx = divUp(W, TX);
	const int by = divUp(H, TY);
	const int bz = divUp(D, TZ);
	const dim3 gridSize(bx,by,bz);
	distanceKernel<<<gridSize,blockSize>>>(d_out, W,H,D,pos);
	cudaMemcpy(out, d_out, W*H*D*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	free(out);
	return 0;
}
