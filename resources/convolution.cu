#define BLOCK_SIZE 32
constexpr int MAX_DIM = 15; //max kernel size 15 x 15

__device__ void mul_vectors(float *a, float *b, float *c, int N)
{
	// int id = blockDim.x * blockIdx.x + threadIdx.x;
	for(int id=0; id < N; id++) c[id] = a[id] * b[id];
}

extern "C" __global__ void convolution(float* lhs, float* kernel, float* out, int W, int H, int W_k, int H_k) 
{
  int stride = 1;
  int nW = W - W_k + 1;
  int nH = H - H_k + 1;

  int ksize = W_k * H_k;
  
  float buffer_lhs[MAX_DIM*MAX_DIM];
  float buffer_mul[MAX_DIM*MAX_DIM];
  float buffer_out[MAX_DIM*MAX_DIM];

  for (int i=0; i<nH; i+=stride) 
  {
    for (int j=0; j<nW; j+=stride) 
	  {
      int idx = i * W + j;
      
      //img2col of a patch
      for (int k=0; k<H_k; k++) {
        memcpy((float*)&buffer_lhs + (k * W_k), lhs + idx + k * W, W_k * sizeof(float));
      }
    
      //perform vector mul and sum
	    mul_vectors((float*)&buffer_lhs, kernel, (float*)&buffer_mul, ksize);
      //sum of mul results
      float sums = 0.0;
      for (int m=0; m<ksize; m++) 
	    {
        sums += buffer_mul[m];
      }
      buffer_out[j] = sums;
    }
   
    //copy to results
	  memcpy(out + i * nW, &buffer_out, nW * sizeof(float));    
  }
}