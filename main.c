#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>


double t1, t2;
int vsize = 0; //vector size;
int m, n = 0; //matrix size;

char data_type; //d - double; f - float
char vect_type[4]; // supported vector instructions: MMX, SSE, AVX, FMA


float *fa, *fa_copy, *fb, *fc;
double *da, *da_copy, *db, *dc;

void init_fvector(float* vector, int size);
void init_dvector(double* vector, int size);
void copy_fvector(float* dst_vector, float* src_vector, int size);
void copy_dvector(double* dst_vector,double* src_vector, int size);

void print_fvector(float* vector, int size);
void print_dvector(double* vector, int size);

//A = B + C
void fvector_summ(float* a, float* b, float* c, int size);
//A = B + C
void fvector_summ_avx(float* a, float* b, float* c, int size);

//A = B + C
void dvector_summ(double* a, double* b, double* c, int size);
//A = B + C
void dvector_summ_avx(double* a, double* b, double* c, int size);




//Vector comparison (without AVX)
int fvector_compare(float* a, float* b, int size);
int dvector_compare(double* a, double* b, int size);


//Utils
void p_init(int argc, char* argv[], int* vsize, char* data_type, char vect_type[]);

int main(int argc, char* argv[])
{
    if (argc == 1)
    {
        printf("No such arguments on input stream. Exit program");
        return 0;
    }
    p_init(argc, argv, &vsize, &data_type, vect_type);
    
    if(strlen(&data_type) == 0) 
    {
        printf("Error: argument with data type (-t) is not passed. Exit program");
        return 0;
    }
    if(vsize == 0)
    {
        printf("Error: argument with data size (-i) is not passed. Exit program");
        return 0;
    }
    
    if(data_type == 'f') 
    {
        fa = (float*) malloc(vsize * sizeof(float));
        fa_copy = (float*) malloc(vsize * sizeof(float));
        fb = (float*) malloc(vsize * sizeof(float));
        fc = (float*) malloc(vsize * sizeof(float));
        init_fvector(fa, vsize);
        init_fvector(fa_copy, vsize);
        init_fvector(fb, vsize);
        init_fvector(fc, vsize);
    }

    if(data_type == 'd') 
    {
        da = (double*) malloc(vsize * sizeof(double));
        da_copy = (double*) malloc(vsize * sizeof(double));
        db = (double*) malloc(vsize * sizeof(double));
        dc = (double*) malloc(vsize * sizeof(double));
        init_dvector(da, vsize);
        init_fvector(da_copy, vsize);
        init_dvector(db, vsize);
        init_dvector(dc, vsize);
    }

    if ((strlen(vect_type) == 0))
    {
        if(data_type == 'f')
        {
            t1 = omp_get_wtime();
            fvector_summ(fa, fb, fc, vsize);
            t2 = omp_get_wtime();

            printf("\nvector_summ\t%c\t%f\t%s", data_type, (t2 - t1), "no_vectorized");

        }
        if(data_type == 'd')
        {
            t1 = omp_get_wtime();
            dvector_summ(da, db, dc, vsize);
            t2 = omp_get_wtime();

            printf("\nvector_summ\t%c\t%f\t%s", data_type, (t2 - t1), "no_vectorized");

        }
    }

    if ((strcmp(vect_type, "AVX") == 0))
    {
        if(data_type == 'f')
        {
            t1 = omp_get_wtime();
            fvector_summ_avx(fa, fb, fc, vsize);
            t2 = omp_get_wtime();

            printf("\nvector_summ\t%c\t%f\t%s", data_type, (t2 - t1), vect_type);

            //Accuracy test
            fvector_summ(fa_copy, fb, fc, vsize);
            if(fvector_compare(fa, fa_copy, vsize) == 1)
                printf("\ncorrect result;");
        }
        if(data_type == 'd')
        {
            t1 = omp_get_wtime();
            dvector_summ_avx(da, db, dc, vsize);
            t2 = omp_get_wtime();

            printf("\nvector_summ\t%c\t%f\t%s", data_type, (t2 - t1), vect_type);

            //Accuracy test
            dvector_summ(da_copy, db, dc, vsize);
            if(dvector_compare(da, da_copy, vsize) == 1)
                printf("\ncorrect result;");
        }
    }

    if(data_type == 'f')
    {
        free(fa);
        free(fb);
        free(fc);
    } 

    if(data_type == 'd')
    {
        free(da);
        free(db);
        free(dc);
    }
    return 0;
}


void init_fvector(float* vector, int size)
{
    for(int i = 0; i < size; i++)
    {
        vector[i] = (i * 1.0);
    }
}

void init_dvector(double* vector, int size)
{
    for(int i = 0; i < size; i++)
    {
        vector[i] = (i * 1.0);
    }  
}

void print_fvector(float* vector, int size)
{
    for(int i = 0; i < size; i++)
    {
        printf("%f ", vector[i]);
    }  
}

void print_dvector(double* vector, int size)
{
    for(int i = 0; i < size; i++)
    {
        printf("%f ", vector[i]);
    }  
}

void fvector_summ(float* a, float* b, float* c, int size)
{
    for(int i = 0; i < size; i++)
    {
        a[i] = b[i] + c[i];
    }
}

void fvector_summ_avx(float* a, float* b, float* c, int size)
{
    __m256 ma, mb, mc;
    int reg_bsize = 32;
    int iter_elements = (reg_bsize / sizeof(float));

    for(int i = 0; i < ((size / iter_elements)  * iter_elements); i += iter_elements)
    {
        mb = _mm256_load_ps(&b[i]);
        mc = _mm256_load_ps(&c[i]);
        ma = _mm256_add_ps(mb, mc);

        _mm256_store_ps(&a[i], ma);
    }

    if((size % iter_elements) == 0)
        return;

    for(int i = ((size / iter_elements)  * iter_elements); i < size; i++)
    {
        a[i] = b[i] + c[i];
    }
}

void dvector_summ(double* a, double* b, double* c, int size)
{
    for(int i = 0; i < size; i++)
    {
        a[i] = b[i] + c[i];
    }
}

void dvector_summ_avx(double* a, double* b, double* c, int size)
{
    __m256d ma, mb, mc;
    int reg_bsize = 32;
    int iter_elements = (reg_bsize / sizeof(double));

    for(int i = 0; i < ((size / iter_elements)  * iter_elements); i += iter_elements)
    {
        mb = _mm256_load_pd(&b[i]);
        mc = _mm256_load_pd(&c[i]);
        ma = _mm256_add_pd(mb, mc);

        _mm256_store_pd(&a[i], ma);
    }

    if((size % iter_elements) == 0)
        return;

    for(int i = ((size / iter_elements)  * iter_elements); i < size; i++)
    {
        a[i] = b[i] + c[i];
    }

}

void copy_fvector(float* dst_vector, float* src_vector, int size)
{
    for(int i = 0; i < size; i++)
    {
        dst_vector[i] = src_vector[i];
    }
}
void copy_dvector(double* dst_vector,double* src_vector, int size)
{
    for(int i = 0; i < size; i++)
    {
        dst_vector[i] = src_vector[i];
    }
}

int fvector_compare(float* a, float* b, int size)
{
    int i = 0;
    
    while ((a[i] == b[i]) && (i < size))
    {
        i++;
    }
    return (i == size) ? 1 : 0; 
}

int dvector_compare(double* a, double* b, int size)
{
    int i = 0;
    
    while ((a[i] == b[i]) && (i < size))
    {
        i++;
    }
    return (i == size) ? 1 : 0; 
}


void p_init(int argc, char* argv[], int* vsize, char* data_type, char vect_type[])
{
    for (int i = 0; i < argc; i++)
    {
        if(strcmp(argv[i], "-i") == 0)
            *vsize = atoi(argv[i + 1]);
        if(strcmp(argv[i], "-t") == 0)
            *data_type = *(argv[i + 1]);
        if(strcmp(argv[i], "-v") == 0)
            strcpy(vect_type, argv[i + 1]);
    }
}