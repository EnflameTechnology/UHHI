
__device__ float sigmoid(float x) {
    return 1.0/(1+expf(-x));
}
__device__ float relu_kernel(float x){return x*(x>0);}
__device__ float elu_kernel(float x){return (x >= 0)*x + (x < 0)*(expf(x)-1);}
__device__ float leaky_kernel(float x){return (x>0) ? x : .1f*x;}
__device__ float tanh_kernel(float x){return (2.f/(1 + expf(-2*x)) - 1);}
__device__ float gelu_kernel(float x) {return x*sigmoid(1.702*x);}

__device__ float activation_kernel(float x, int act)
{
    switch(act){
        case 0:
            return relu_kernel(x);
        case 1:
            return gelu_kernel(x);
        case 2:
            return leaky_kernel(x);
        case 3:
            return tanh_kernel(x);
    }
    return 0;
}

extern "C"  __global__ void activation(float *x, int N, int act)
{

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    // float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        int i = ROW * N + COL;
        if(i < N * N) x[i] = activation_kernel(x[i], act);
    }

    // int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    
}