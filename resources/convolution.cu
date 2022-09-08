#define BLOCK_SIZE 32
extern "C" __global__ void convolution(float* A, float* B, float* C, int HA, int WA, int HB, int WB, int HC, int WC)
{
	int col = blockIdx.x * (BLOCK_SIZE - WC + 1) + threadIdx.x;
	int row = blockIdx.y * (BLOCK_SIZE - WC + 1) + threadIdx.y;
	int row_i = row - WC + 1;
	int col_i = col - WC + 1;

	float tmp = 0;

	__shared__ float shm[BLOCK_SIZE][BLOCK_SIZE];

	if (row_i < WA && row_i >= 0 && col_i < WA && col_i >= 0)
	{
		shm[threadIdx.y][threadIdx.x] = A[col_i * WA + row_i];
	}
	else
	{
		shm[threadIdx.y][threadIdx.x] = 0;
	}

	__syncthreads();

	if (threadIdx.y < (BLOCK_SIZE - WC + 1) && threadIdx.x < (BLOCK_SIZE - WC + 1) && row < (WB - WC + 1) && col < (WB - WC + 1))
	{
		for (int i = 0; i< WC;i++)
			for (int j = 0;j<WC;j++)
				tmp += shm[threadIdx.y + i][threadIdx.x + j] * C[j*WC + i];
		B[col*WB + row] = tmp;
	}
}