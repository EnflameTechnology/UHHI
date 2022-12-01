extern "C" __global__ void elementi32(int* mat_a, int* mat_b, int* mat_out, unsigned int rows, unsigned int cols, unsigned int type) 
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) 
    {
        unsigned int pos = idy * cols + idx;

        if (type == 0) {
            mat_out[pos] = mat_a[pos] + mat_b[pos];
        } else if (type == 1) {
            mat_out[pos] = mat_a[pos] - mat_b[pos];
        } else if (type == 2) {
            mat_out[pos] = mat_a[pos] * mat_b[pos];
        } else if (type == 3) {
            mat_out[pos] = int(mat_a[pos] / mat_b[pos]);
        }

    }
}