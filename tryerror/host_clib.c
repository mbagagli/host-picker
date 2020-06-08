// gcc -shared -Wl,-soname,host_clib -O3 -o host_clib.so -fPIC host_clib.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ===================================  Declare Functions

int kurtcf(float* arr, int szarr, int szwin, /*@out@*/ float* kcf);
int kurtcf_mean(float* arr, int szarr, int szwin, /*@out@*/ float* kcf);

int skewcf(float* arr, int szarr, int szwin, /*@out@*/ float* scf);
int skewcf_mean(float* arr, int szarr, int szwin, /*@out@*/ float* scf);

/* ------------  MyHELP
int A[5]
Address: &A[i] OR (A+i)
Value:    A[i]  OR *(A+i)

%p requires an argument of type void*, not just of any pointer type.
If you want to print a pointer value, you should cast it to void*:
printf("&q = %p\n", (void*)&q);

https://www.youtube.com/watch?v=CpjVucvAc3g
-------------- */




/* ---------------------------------------------
                    KURTOSIS
   --------------------------------------------- */

// The next one is a simplified definition of KURTOSIS:
//  - the input assume a ZERO-MEAN process! (filtered and deameaned)
int kurtcf(float* arr, int szarr, int szwin, /*@out@*/ float* kcf)
{
    // Declare MAIN
    int k;  // MAIN loop
    memset(kcf, 0, szarr*sizeof(float));

    int x; // AUX loop
    float mom4, mom2;

    // Work
    for (k=szwin; k<szarr; k++) {
        mom4 = 0.0;
        mom2 = 0.0;
        for (x=k-szwin; x<=k; x++){
            mom4 += pow(arr[x], 4);
            mom2 += pow(arr[x], 2);
        }

        mom4 = mom4 / szwin;
        mom2 = mom2 / szwin;
        kcf[k] = mom4 / pow(mom2, 2);
    }
    //
    return 0;
}

// The next one is the classic definition of KURTOSIS:
//  - the input could be a NON-ZERO mean process
int kurtcf_mean(float* arr, int szarr, int szwin, /*@out@*/ float* kcf)
{
    // Declare MAIN
    int k;  // MAIN loop
    memset(kcf, 0, szarr*sizeof(float));

    int x, _x; // AUX loop
    float mom4, mom2, mean, summ;

    // Work
    for (k=szwin; k<szarr; k++) {
        mom4 = 0.0;
        mom2 = 0.0;

        for (x=k-szwin; x<=k; x++){

            summ = 0.0;
            mean = 0.0;
            for (_x=x-szwin; _x<=x; _x++){
                summ += arr[abs(_x)];
            }
            mean = summ / szwin;

            mom4 += pow((arr[x] - mean), 4);
            mom2 += pow((arr[x] - mean), 2);
        }
        mom4 = mom4 / szwin;
        mom2 = mom2 / szwin;
        kcf[k] = mom4 / pow(mom2, 2);
    }
    //
    return 0;
}




/* ---------------------------------------------
                    SKEWNESS
   --------------------------------------------- */

// The next one is a simplified definition of SKEWNESS:
//  - the input assume a ZERO-MEAN process! (filtered and deameaned)
int skewcf(float* arr, int szarr, int szwin, /*@out@*/ float* scf)
{
    // Declare MAIN
    int k;  // MAIN loop
    memset(scf, 0, szarr*sizeof(float));

    int x; // AUX loop
    float mom3, mom2;

    // Work
    for (k=szwin; k<szarr; k++) {
        mom3 = 0.0;
        mom2 = 0.0;
        for (x=k-szwin; x<=k; x++){
            mom3 += pow(arr[x], 3);
            mom2 += pow(arr[x], 2);
        }

        mom3 = mom3 / szwin;
        mom2 = mom2 / szwin;
        scf[k] = mom3 / sqrt(pow(mom2, 3));
    }
    //
    return 0;
}

// The next one is the classic definition of SKEWNESS:
//  - the input could be a NON-ZERO mean process
int skewcf_mean(float* arr, int szarr, int szwin, /*@out@*/ float* scf)
{
    // Declare MAIN
    int k;  // MAIN loop
    memset(scf, 0, szarr*sizeof(float));

    int x, _x; // AUX loop
    float mom3, mom2, mean, summ;

    // Work
    for (k=szwin; k<szarr; k++) {
        mom3 = 0.0;
        mom2 = 0.0;

        for (x=k-szwin; x<=k; x++){

            summ = 0.0;
            mean = 0.0;
            for (_x=x-szwin; _x<=x; _x++){
                summ += arr[abs(_x)];
            }
            mean = summ / szwin;

            mom3 += pow((arr[x] - mean), 3);
            mom2 += pow((arr[x] - mean), 2);
        }
        mom3 = mom3 / szwin;
        mom2 = mom2 / szwin;
        scf[k] = mom3 / sqrt(pow(mom2, 3));
    }
    //
    return 0;
}
