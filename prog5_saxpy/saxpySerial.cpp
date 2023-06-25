#include <immintrin.h>


void saxpySerial(int N,
                 float scale,
                 float X[],
                 float Y[],
                 float result[]) {

    for (int i = 0; i < N; i++) {
        result[i] = scale * X[i] + Y[i];
    }
}

void saxpyImproved(int N,
                   float scale,
                   float *__restrict__ X,
                   float *__restrict__ Y,
                   float *__restrict__ result) {
    constexpr int vector_dim = 8;

    constexpr int vector_dim_1 = vector_dim;
    constexpr int vector_dim_2 = vector_dim * 2;
    constexpr int vector_dim_3 = vector_dim * 3;
    constexpr int vector_dim_4 = vector_dim * 4;
    constexpr int vector_dim_5 = vector_dim * 5;
    constexpr int vector_dim_6 = vector_dim * 6;
    constexpr int vector_dim_7 = vector_dim * 7;
    constexpr int block_size = vector_dim * 8;

    const int N_vec_unroll = (N / block_size) * block_size;
    const int N_rest_unroll = N - N_vec_unroll;

    __m256 scale_v = _mm256_set1_ps(scale);

    for (int i = 0; i < N_vec_unroll; i += block_size) {
        const float *x_ptr = X + i;
        __m256 x1 = _mm256_loadu_ps(x_ptr);
        __m256 x2 = _mm256_loadu_ps(x_ptr + vector_dim_1);
        __m256 x3 = _mm256_loadu_ps(x_ptr + vector_dim_2);
        __m256 x4 = _mm256_loadu_ps(x_ptr + vector_dim_3);
        __m256 x5 = _mm256_loadu_ps(x_ptr + vector_dim_4);
        __m256 x6 = _mm256_loadu_ps(x_ptr + vector_dim_5);
        __m256 x7 = _mm256_loadu_ps(x_ptr + vector_dim_6);
        __m256 x8 = _mm256_loadu_ps(x_ptr + vector_dim_7);

        const float *y_ptr = Y + i;
        __m256 y1 = _mm256_loadu_ps(y_ptr);
        __m256 y2 = _mm256_loadu_ps(y_ptr + vector_dim_1);
        __m256 y3 = _mm256_loadu_ps(y_ptr + vector_dim_2);
        __m256 y4 = _mm256_loadu_ps(y_ptr + vector_dim_3);
        __m256 y5 = _mm256_loadu_ps(y_ptr + vector_dim_4);
        __m256 y6 = _mm256_loadu_ps(y_ptr + vector_dim_5);
        __m256 y7 = _mm256_loadu_ps(y_ptr + vector_dim_6);
        __m256 y8 = _mm256_loadu_ps(y_ptr + vector_dim_7);

        float *result_ptr = result + i;
        _mm256_stream_ps(result_ptr, _mm256_fmadd_ps(scale_v, x1, y1));
        _mm256_stream_ps(result_ptr + vector_dim_1, _mm256_fmadd_ps(scale_v, x2, y2));
        _mm256_stream_ps(result_ptr + vector_dim_2, _mm256_fmadd_ps(scale_v, x3, y3));
        _mm256_stream_ps(result_ptr + vector_dim_3, _mm256_fmadd_ps(scale_v, x4, y4));
        _mm256_stream_ps(result_ptr + vector_dim_4, _mm256_fmadd_ps(scale_v, x5, y5));
        _mm256_stream_ps(result_ptr + vector_dim_5, _mm256_fmadd_ps(scale_v, x6, y6));
        _mm256_stream_ps(result_ptr + vector_dim_6, _mm256_fmadd_ps(scale_v, x7, y7));
        _mm256_stream_ps(result_ptr + vector_dim_7, _mm256_fmadd_ps(scale_v, x8, y8));
    }

    const int N_vec = N_vec_unroll + ((N_rest_unroll / vector_dim) * vector_dim);
    const int N_rest = N - N_vec;

    for (int i = N_vec_unroll; i < N_vec; i += vector_dim) {
        __m256 x1 = _mm256_loadu_ps(X + i);
        __m256 y1 = _mm256_loadu_ps(Y + i);

        _mm256_stream_ps(result + i, _mm256_fmadd_ps(scale_v, x1, y1));
    }

    saxpySerial(N_rest, scale, X + N_vec, Y + N_vec, result);

}

