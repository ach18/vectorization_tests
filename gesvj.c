#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mkl.h"
#include <immintrin.h>
#include <omp.h>

double t1, t2;
int m, n = 0; //matrix size;

char data_type; //d - double; f - float
char vect_type[4]; // supported vector instructions: MMX, SSE, AVX, FMA


float *fa, *fa_copy, *fu, *fv, *fs;
double *da, *da_copy, *du, *dv, *ds;
double* dwork; //for MKL dgesvj

void finit_rand_matr(float* matr, int m, int n); //+
void finit_ident_matr(float* matr, int m, int n); //+
void fcopy_matr(float* dest_matr, float* src_matr, int m, int n); //+
void dinit_rand_matr(double* matr, int m, int n); //+
void dinit_ident_matr(double* matr, int m, int n);//+
void dcopy_matr(double* dest_matr, double* src_matr, int m, int n);//+
void dtransp_matr(double* dest_matr, double* src_matr, int m, int n); 

//Matrix comparison (without eps tol)
int fmatr_compare(float* matr_a, float* matr_b, int m, int n);
int dmatr_compare(double* matr_a, double* matr_b, int m, int n);

//plane jacobi (without blocking)
void dgesvj_nb(double* amatr, double* umatr, double* vmatr, double* svect, int m, int n);

//block jacobi
void dgesvj_b(double* amatr, double* umatr, double* vmatr, double* svect, int m, int n);

//Utils
void p_init(int argc, char* argv[], int* m, int* n, char* data_type, char vect_type[]);
void dfrobenius(double* amatr, int m, int n, double* norm, double* off_norm);
void dsort_vals(double* umatr, double* vmatr, double* svect, int m, int n);
void dprint_matr(double* amatr, int m, int n);


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
        fv = (float*) malloc(n * n * sizeof(float));
        fs = (float*) malloc(n * sizeof(float));

        finit_rand_matr(fa, m, n);
        fcopy_matr(fa_copy, fa, m, n);
        finit_ident_matr(fu, m, n);
        finit_ident_matr(fv, n, n);
    }

    if(data_type == 'd') 
    {
        da = (double*) malloc(m * n * sizeof(double));
        da_copy = (double*) malloc(m * n * sizeof(double));
        du = (double*) malloc(m * n * sizeof(double));
        dv = (double*) malloc(n * n * sizeof(double));
        ds = (double*) malloc(n * sizeof(double));
        dinit_rand_matr(da, m, n);
        dcopy_matr(da_copy, da, m, n);
        dinit_ident_matr(du, m, n);
        dinit_ident_matr(dv, n, n);
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
            //dgesvj_nb(da_copy, du, dv, ds, m, n);
            dgesvj_b(da_copy, du, dv, ds, m, n);
            t2 = omp_get_wtime();
            return;

            printf("\ngesvj\t%c\t%f\t%s", data_type, (t2 - t1), "no_vectorized");
            printf("\n");
            dprint_matr(ds, n, 1);
            //dprint_matr(da_copy, m, n);


            //gesvj MKL
            dcopy_matr(da_copy, da, m, n);
            dinit_ident_matr(du, m, n);
            dinit_ident_matr(dv, n, n);

            char joba = 'G';
            char jobu = 'U';
            char jobv = 'V';

            MKL_INT lda = m;
            MKL_INT ldv = n;
            MKL_INT mv = 0;
            MKL_INT lwork = m + n;
            MKL_INT dgesvj_info = -1;
            dwork = (double*)malloc(lwork * sizeof(double));


            t1 = omp_get_wtime();
            dgesvj(&joba, &jobu, &jobv, &m, &n, da_copy, &lda, ds, &mv, dv, &ldv, dwork, &lwork, &dgesvj_info);
            t2 = omp_get_wtime();

            printf("\ngesvj\t%c\t%f\t%s", data_type, (t2 - t1), "MKL");
            printf("\n");
            dprint_matr(ds, n, 1);
            //stores U in V!!!
            //dprint_matr(dv, n, n);
            free(dwork);
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
            strcpy_s(vect_type, 4, argv[i + 1]);
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

void dfrobenius(double* amatr, int m, int n, double* norm, double* off_norm)
{
    double sum = 0.0;       // sum m[i][j]^2 for 0 < i < m and 0 < j < n
    double off_sum = 0.0;  // sum m[i][j]^2 for 0 < i < m and 0 < j < n and i == j

    //double LAPACKE_dlange
    //sum = LAPACKE_dlange(matrix_layout, 'F', m, n, data, m);

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            sum += amatr[(n * i) + j] * amatr[(n * i) + j];
            if(i == j)
                off_sum += amatr[(n * i) + j] * amatr[(n * i) + j];
        }
    }
    *norm = sum;
    *off_norm = sum - off_sum;
}

void dsort_vals(double* umatr, double* vmatr, double* svect, int m, int n)
{
    for (int i = 0; i < n; i++) 
    {
        double s_last = svect[i];
        int i_last = i;
        for (int j = i + 1; j < n; j++) 
        {
            if (svect[j] > s_last) 
            {
                s_last = svect[j];
                i_last = j;
            }
        }
        if (i_last != i) 
        {
            double tmp;
            tmp = svect[i];
            svect[i] = svect[i_last];
            svect[i_last] = tmp;

            for (int k = 0; k < m; k++) 
            {
                tmp = umatr[(k * n) + i];
                umatr[(k * n) + i] = umatr[(k * n) + i_last];
                umatr[(k * n) + i_last] = tmp;
            }
            for (int k = 0; k < m; k++)
            {
                tmp = vmatr[(k * n) + i];
                vmatr[(k * n) + i] = vmatr[(k * n) + i_last];
                vmatr[(k * n) + i_last] = tmp;
            }
        }
    }
}

void dprint_matr(double* amatr, int m, int n)
{
    //for (int i = 0; i < m; i++)
    //{
    //    for (int j = 0; j < n; j++)
    //    {
    //        printf("%f\t", amatr[(j*n) + i]);
    //    }
    //    printf("\n");
    //}
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.1f\t", amatr[(i * n) + j]);
        }
        printf("\n");
    }
}

//testing...
void dgesvj_nb(double* amatr, double* umatr, double* vmatr, double* svect, int m, int n)
{
    int iter = 0;
    int max_sweeps = 500;
    int converged = 0;
    double tol = 1e-15;
    double norm = 0.0;
    double off_norm = 0.0;

    double bii = 0.0;
    double bij = 0.0;
    double bji = 0.0;
    double bjj = 0.0;
    double tau = 0.0;
    double t = 0.0;
    double c = 0.0;
    double s = 0.0;
    //double *acopy_matr = (double*) malloc(m * n * sizeof(double));
    double *gramm_matr = (double*) malloc(m * n * sizeof(double));

    //cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, n, 1.0, amatr, m, amatr, n, 0.0, gramm_matr, n);
    //dfrobenius(amatr, m, n, &norm, &off_norm);
    //while (sqrt(off_norm) > tol * sqrt(norm))
    //while (sqrt(off_norm) > tol)
    do
    {
        converged = 1;
        //B^T * B
        //cblas_dcopy(m * n, amatr, 1, acopy_matr, 1);
        //is this correct operation for m > n?
        //cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, n, 1.0, amatr, m, amatr, n, 0.0, gramm_matr, n);

        //n or m?
        for (int i = 0; i < (n - 1); i++)
        {
            for (int j = (i + 1); j < n; j++)
            {
                bii = 0.0; bij = 0.0; bjj = 0.0;

                //n or m? 
                //!upper triangle elements
                //bii = gramm_matr[n * i + i];
                //bij = gramm_matr[n * i + j];
                ////bji = gramm_matr[n * j + i];
                //bjj = gramm_matr[n * j + j];

                //Gramm matrix computation - multiply of columns?
                for (int k = 0; k < m; k++)
                {
                    bii += amatr[n * k + i] * amatr[n * k + i];
                    //is this correct operation?
                    bij += amatr[n * k + i] * amatr[n * k + j];
                    bjj += amatr[n * k + j] * amatr[n * k + j];
                }

                if (abs(bij) >= (tol * sqrt(bii * bjj)))
                {
                    converged = 0;

                    //if (bij != 0.0)
                    //{
                        tau = (bjj - bii) / (2.0 * bij);
                        double sign = tau > 0.0 ? 1.0 : -1.0;
                        t = sign / (abs(tau) + sqrt(1.0 + (tau * tau)));
                        //if(tau >= 0)
                        //    t = 1.0 / (tau + sqrt(1.0 + (tau * tau)));
                        //else 
                        //    t = -1.0 / (-tau + sqrt(1.0 + (tau * tau)));

                        c = 1.0 / sqrt(1.0 + (t * t));
                        s = t * c;
                    //}
                    /*else
                    {
                        c = 1.0;
                        s = 0.0;
                    }*/

                    // n or m? of columns m +
                    for (int k = 0; k < m; k++)
                    {
                        double b_ki = amatr[n * k + i];
                        double b_kj = amatr[n * k + j];

                        double left = (c * b_ki) - (s * b_kj);
                        double right = (s * b_ki) + (c * b_kj);

                        amatr[n * k + i] = left;
                        amatr[n * k + j] = right;
                    }

                    //n or m? V is n x n +
                    for (int k = 0; k < n; k++)
                    {
                        double v_ki = vmatr[n * k + i];
                        double v_kj = vmatr[n * k + j];

                        double left = (c * v_ki) - (s * v_kj);
                        double right = (s * v_ki) + (c * v_kj);

                        vmatr[n * k + i] = left;
                        vmatr[n * k + j] = right;
                    }
                }
            }
        }
        //cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, n, 1.0, amatr, m, amatr, n, 0.0, gramm_matr, n);
        //dfrobenius(amatr, m, n, &norm, &off_norm);
        //if (iter > max_sweeps)
        //{
        //    printf("\nSWEEPS %d", iter);
        //    return;
        //}
        //else
        iter++;
        //converges criteria?
        //} while (iter < 100);
    } while (!converged);

    //n or m?
    for (int i = 0; i < n; i++) 
    {
        double sigma = 0.0;

        // ||Ai||2 +
        for (int k = 0; k < m; k++) 
        {
            sigma += amatr[n * k + i] * amatr[n * k + i];
        }
        sigma = sqrt(sigma);

        svect[i] = sigma;

        //U - zeroes-like ?
        for (int k = 0; k < m; k++) 
        {
            amatr[n * k + i] /= sigma;
        }
    }
    //printf("\nEND");
    dsort_vals(amatr, vmatr, svect, m, n);
    free(gramm_matr);
}

void dgesvj_b(double* amatr, double* umatr, double* vmatr, double* svect, int m, int n)
{
    int iter = 0;
    int max_sweeps = 500;
    int converged = 0;
    int p = 1; //num of CPUs
    int l = 2 * p; // num of blocks
    int k = n / l; // block size
    double tol = 1e-15;
    double norm = 0.0;
    double off_norm = 0.0;

    double* acopy_matr = (double*)malloc(m * n * sizeof(double));

    double* al_matr = (double*)malloc(m * k * sizeof(double));
    double* ar_matr = (double*)malloc(m * k * sizeof(double));

    double* tmp_matr = (double*)malloc(k * k * sizeof(double));

    double* gramm_matr = (double*)malloc(k * k * 4 * sizeof(double)); // G = (G_ll G_lr G^T_lr G_rr)
    //double* gramm_matr = (double*)malloc(m * k * 4 * sizeof(double)); //temporary size for test

    cblas_dcopy(m * n, amatr, 1, acopy_matr, 1);

    dfrobenius(acopy_matr, m, n, &norm, &off_norm);
    //while (sqrt(off_norm) > tol * sqrt(norm))
    //while (sqrt(off_norm) >= tol)
    //{
        //copy to A_l
        for (int i = 0; i < m; i++)
        {
            cblas_dcopy(k, &acopy_matr[n * i], 1, &al_matr[k * i], 1);
        }
        //copy to A_r
        for (int i = 0; i < m; i++)
        {
            cblas_dcopy(k, &acopy_matr[(n * i) + (n-k)], 1, &ar_matr[k * i], 1);
        }

        //A^T_l * A_l
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, k, k, m, 1.0, al_matr, k, al_matr, k, 0.0, tmp_matr, k);
        //copy (A^T_l * A_l) to G_ll
        for (int i = 0; i < k; i++)
        {
            cblas_dcopy(k, &tmp_matr[k * i], 1, &gramm_matr[2 * k * i], 1);
        }
        //printf("\n");
        //dprint_matr(tmp_matr, k, k);

        //A^T_l * A_r
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, k, k, m, 1.0, al_matr, k, ar_matr, k, 0.0, tmp_matr, k);
        //copy (A^T_l * A_r) to G_lr
        for (int i = 0; i < k; i++)
        {
            cblas_dcopy(k, &tmp_matr[k * i], 1, &gramm_matr[(2 * k * i) + ((2 * k) - k)], 1);
        }
        //printf("\n");
        //dprint_matr(tmp_matr, k, k);


        //A^T_r * A_l
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, k, k, m, 1.0, ar_matr, k, al_matr, k, 0.0, tmp_matr, k);
        //copy (A^T_r * A_l) to G^T_lr
        for (int i = 0; i < k; i++)
        {
            cblas_dcopy(k, &tmp_matr[k * i], 1, &gramm_matr[(2 * k) * (k + i)], 1);
        }
        //printf("\n");
        //dprint_matr(tmp_matr, k, k);

        //A^T_r * A_r
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, k, k, m, 1.0, ar_matr, k, ar_matr, k, 0.0, tmp_matr, k);
        //copy (A^T_r * A_r) to G_rr
        for (int i = 0; i < k; i++)
        {
            cblas_dcopy(k, &tmp_matr[k * i], 1, &gramm_matr[((2 * k) * (k + i)) + k], 1);
        }
        //dprint_matr(al_matr, m, k);
        //printf("\n");
        //dprint_matr(tmp_matr, k, k);
        //dprint_matr(acopy_matr, m, n);
        //printf("\n");
        //dprint_matr(gramm_matr, 2 * k, 2 * k);
        //dprint_matr(al_matr, m, k);
        //printf("\n");
        //dprint_matr(ar_matr, m, k);
        
    //}

}