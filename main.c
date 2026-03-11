#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>


double t1, t2;
int vsize = 0;
char data_type; //d - double; f - float
char vect_type[4]; // supported vector instructions: MMX, SSE, SSE2, SSE3, AVX, AVX2, FMA3


float *fa, *fb, *fc;
double *da, *db, *dc;

void init_fvector(float* vector, int size);
void init_dvector(double* vector, int size);
void print_fvector(float* vector, int size);
void print_dvector(double* vector, int size);

void fvector_summ(float* a, float* b, float* c, int size);
void dvector_summ(double* a, double* b, double* c, int size);

int main(int argc, char* argv[])
{
    if (argc == 1)
    {
        printf("No such arguments on input stream. Exit program");
        return 0;
    }

    for (int i = 0; i < argc; i++)
    {
        if(strcmp(argv[i], "-i") == 0)
            vsize = atoi(argv[i + 1]);
        if(strcmp(argv[i], "-t") == 0)
            data_type = *(argv[i + 1]);
        if(strcmp(argv[i], "-v") == 0)
            strcpy(vect_type, argv[i + 1]);
    }
    
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
        fb = (float*) malloc(vsize * sizeof(float));
        fc = (float*) malloc(vsize * sizeof(float));
        init_fvector(fa, vsize);
        init_fvector(fb, vsize);
        init_fvector(fc, vsize);
    }


    if(data_type == 'd') 
    {
        da = (double*) malloc(vsize * sizeof(double));
        db = (double*) malloc(vsize * sizeof(double));
        dc = (double*) malloc(vsize * sizeof(double));
        init_dvector(da, vsize);
        init_dvector(db, vsize);
        init_dvector(dc, vsize);
    }

    if ((strlen(vect_type) == 0) && (data_type == 'f'))
    {
        t1 = omp_get_wtime();
        fvector_summ(fa, fb, fc, vsize);
        t2 = omp_get_wtime();

        printf("\nvector_summ\t%c\t%f", data_type, (t2 - t1));
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
    
    //printf("vsize = %d ; and arg is %s", vsize, argv[1]);
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

void dvector_summ(double* a, double* b, double* c, int size)
{
    for(int i = 0; i < size; i++)
    {
        a[i] = b[i] + c[i];
    }
}