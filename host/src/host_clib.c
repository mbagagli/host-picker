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

int aicp(float* arr, int sz, /*@out@*/ float* aic, int* minidx);
// int aicp(float* arr, int sz, /*@out@*/ float* aic, int* pminidx);

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
    // printf("Using: KURTOSIS\n");
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
    // printf("Using: KURTOSIS-MEAN\n");
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
    // printf("Using: SKEWNESS\n");
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
    // printf("Using: SKEWNESS-MEAN\n");
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



/* ---------------------------------------------
        AkaikeInformationCriteria (AIC)
   --------------------------------------------- */

int aicp(float* arr, int sz, /*@out@*/ float* aic, int* pminidx) {

    // Declare MAIN
    int ii;  // MAIN loop
    int minidx = 0;
    float minval = INFINITY;
    //float var1, var2, val1, val2;
    memset(aic, 0, (sz-1)*sizeof(float));

    // Declare VARIANCE
    //float sd;
    int _x, _xx, _y, _yy; // AUX loop
    float sumOne, meanOne;
    float sumTwo, meanTwo;
    float devOne, sdevOne;
    float devTwo, sdevTwo;
    //
    float valOne, valTwo;


    // Work
    for (ii=1; ii<sz; ii++) {

        // Loop for VAR 1
        sumOne = 0.0;
        for (_x=0; _x<ii; _x++){
            sumOne = sumOne + arr[_x];
        }
        meanOne = sumOne / ii;

        sdevOne = 0.0;
        for (_xx=0; _xx<ii; _xx++){
            devOne = (arr[_xx] - meanOne) * (arr[_xx] - meanOne);
            sdevOne = sdevOne + devOne;
        }
        //var1 = sdevOne / ii;
        //sd = sqrt(var1);
        valOne = ii * log(sdevOne / ii);



        // Loop for VAR 2
        sumTwo = 0.0;
        for (_y=ii; _y<sz; _y++){
            sumTwo = sumTwo + arr[_y];
        }
        meanTwo = sumTwo / (sz - ii);

        sdevTwo = 0.0;
        for (_yy=ii; _yy<sz; _yy++){
            devTwo = (arr[_yy] - meanTwo) * (arr[_yy] - meanTwo);
            sdevTwo = sdevTwo + devTwo;
        }
        //var2 = sdevTwo / (sz - ii);
        //sd = sqrt(var2);
        valTwo = (sz - ii - 1) * log(sdevTwo / (sz - ii));

        // Allocate to AIC
        aic[ii - 1] = (valOne + valTwo);

        // Find MINIMA

        if ( isinf(aic[ii-1]) ) {
            aic[ii-1] = INFINITY;
        }

        if ( isnan(aic[ii-1]) ) {
            aic[ii-1] = INFINITY;
        }


        // Not minor equal, but just minor
        if (aic[ii-1] < minval) {
            minval = aic[ii-1];
            minidx = ii-1;
        }
    }
    //
    *pminidx = minidx;  // return IDX
    return 0;
}
