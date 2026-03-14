#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

double t1, t2;
int m, n = 0; //matrix size;

char data_type; //d - double; f - float
char vect_type[4]; // supported vector instructions: MMX, SSE, AVX, FMA


float *fa, *fa_copy, *fu, *fv;
double *da, *da_copy, *du, *dv;

void finit_rand_matr(float* matr, int m, int n); //+
void finit_ident_matr(float* matr, int m, int n); //+
void fcopy_matr(float* dest_matr, float* src_matr, int m, int n); //+
void dinit_rand_matr(double* matr, int m, int n); //+
void dinit_ident_matr(double* matr, int m, int n);//+
void dcopy_matr(double* dest_matr, double* src_matr, int m, int n);

//Matrix comparison (without eps tol)
int fmatr_compare(float* matr_a, float* matr_b, int m, int n);
int dmatr_compare(double* matr_a, double* matr_b, int m, int n);


//Utils
void p_init(int argc, char* argv[], int* m, int* n, char* data_type, char vect_type[]);

int main(int argc, char* argv[])
{
    if (argc == 1)
    {
        printf("No such arguments on input stream. Exit program");
        return 0;
    }
    p_init(argc, argv, &m, &n, &data_type, vect_type);
    
    if(strlen(&data_type) == 0) 
    {
        printf("Error: argument with data type (-t) is not passed. Exit program");
        return 0;
    }
    if(m == 0)
    {
        printf("Error: argument with row matrix size (-m) is not passed. Exit program");
        return 0;
    }
    if(n == 0)
    {
        printf("Error: argument with column matrix size (-n) is not passed. Exit program");
        return 0;
    }

    if(data_type == 'f') 
    {
        fa = (float*) malloc(m * n * sizeof(float));
        fa_copy = (float*) malloc(m * n * sizeof(float));
        fu = (float*) malloc(m * n * sizeof(float));
        fv = (float*) malloc(m * n * sizeof(float));
        finit_rand_matr(fa, m, n);
        fcopy_matr(fa_copy, fa, m, n);
        finit_ident_matr(fu, m, n);
        finit_ident_matr(fv, m, n);
    }

    if(data_type == 'd') 
    {
        da = (double*) malloc(m * n * sizeof(double));
        da_copy = (double*) malloc(m * n * sizeof(double));
        du = (double*) malloc(m * n * sizeof(double));
        dv = (double*) malloc(m * n * sizeof(double));
        dinit_rand_matr(da, m, n);
        dcopy_matr(da_copy, da, m, n);
        dinit_ident_matr(du, m, n);
        dinit_ident_matr(dv, m, n);
    }

    if ((strlen(vect_type) == 0))
    {
        if(data_type == 'f')
        {
            t1 = omp_get_wtime();
            //fvector_summ(fa, fb, fc, vsize);
            t2 = omp_get_wtime();

            printf("\ngesvj\t%c\t%f\t%s", data_type, (t2 - t1), "no_vectorized");

        }
        if(data_type == 'd')
        {
            t1 = omp_get_wtime();
            //dvector_summ(da, db, dc, vsize);
            t2 = omp_get_wtime();

            printf("\ngesvj\t%c\t%f\t%s", data_type, (t2 - t1), "no_vectorized");

        }
    }

    if(data_type == 'f')
    {
        free(fa);
        free(fa_copy);
        free(fu);
        free(fv);
    } 

    if(data_type == 'd')
    {
        free(da);
        free(da_copy);
        free(du);
        free(dv);
    }
    return 0;
}


//Utils
void p_init(int argc, char* argv[], int* m, int* n, char* data_type, char vect_type[])
{
    for (int i = 0; i < argc; i++)
    {
        if(strcmp(argv[i], "-m") == 0)
            *m = atoi(argv[i + 1]);
        if(strcmp(argv[i], "-n") == 0)
            *n = atoi(argv[i + 1]);
        if(strcmp(argv[i], "-t") == 0)
            *data_type = *(argv[i + 1]);
        if(strcmp(argv[i], "-v") == 0)
            strcpy(vect_type, argv[i + 1]);
    }
}

//Matrix init
void finit_rand_matr(float* matr, int m, int n)
{
    for(int i = 0; i < (m * n); i++)
    {
        matr[i] = (rand() % 100) * 1.0;
    }
}

void dinit_rand_matr(double* matr, int m, int n)
{
    for(int i = 0; i < (m * n); i++)
    {
        matr[i] = (rand() % 100) * 1.0;
    }
}

void finit_ident_matr(float* matr, int m, int n)
{
    int ind;
    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            ind = i * n + j;
            //if (Pmat.storage = 'C')
            //    index = j * N + i;

            if (i == j) 
            {
                matr[ind] = 1.0;
            }
            else 
            {
                matr[ind] = 0.0;
            }
        }
    }
}

void dinit_ident_matr(double* matr, int m, int n)
{
    int ind;
    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            ind = i * n + j;
            //if (Pmat.storage = 'C')
            //    index = j * N + i;

            if (i == j) 
            {
                matr[ind] = 1.0;
            }
            else 
            {
                matr[ind] = 0.0;
            }
        }
    }
}

void fcopy_matr(float* dest_matr, float* src_matr, int m, int n)
{
    //copy like a vector
    for(int i = 0; i < (m * n); i++)
    {
        dest_matr[i] = src_matr[i];
    }
}

void dcopy_matr(double* dest_matr, double* src_matr, int m, int n)
{
    //copy like a vector
    for(int i = 0; i < (m * n); i++)
    {
        dest_matr[i] = src_matr[i];
    }
}