/*=================================================================
 *
 * YPRIME.C	Sample .MEX file corresponding to YPRIME.M
 *	        Solves simple 3 body orbit problem 
 *
 * The calling syntax is:
 *
 *		[yp] = yprime(t, y)
 *
 *  You may also want to look at the corresponding M-code, yprime.m.
 *
 * This is a MEX-file for MATLAB.  
 * Copyright 1984-2011 The MathWorks, Inc.
 *
 *=================================================================*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if !defined(MAX)
#define	MAX(A, B)	((A) > (B) ? (A) : (B))
#endif

#if !defined(MIN)
#define	MIN(A, B)	((A) < (B) ? (A) : (B))
#endif

static	double	mu = 1/82.45;
static	double	mus = 1 - 1/82.45;

/////////////////////////////////////////////////////////////////////////
//===================== Method 1: =============================================
//Algorithm from N. Wirth’s book Algorithms + data structures = programs of 1976  
// typedef int elem_type ;
#define ELEM_SWAP(a,b) { register double t=(a);(a)=(b);(b)=t; }
double kth_smallest(double a[], int n, int k)
{
    int i,j,l,m ;
    double x ;
    l=0 ; m=n-1 ;
    while (l<m) {
    x=a[k] ;
    i=l ;
    j=m ;
    do {
    while (a[i]<x) i++ ;
    while (x<a[j]) j-- ;
    if (i<=j) {
    ELEM_SWAP(a[i],a[j]) ;
    i++ ; j-- ;
    }
    } while (i<=j) ;
    if (j<k) l=i ;
    if (k<i) m=j ;
    }
    return a[k] ;
}
#define wirth_median(a,n) kth_smallest(a,n,(((n)&1)?((n)/2):(((n)/2)-1)))

//===================== Method 2: =============================================
//This is the faster median determination method.
//Algorithm from Numerical recipes in C of 1992
double quick_select_median(double arr[], int n)
{
    int low, high ;
    int median;
    int middle, ll, hh;
    low = 0 ; high = n-1 ; median = (low + high) / 2;
//     printf("\n n=%d",n);
//      printf("\n low=%d",low); printf("\t high=%d",high); printf("\t median=%d",median);
//     printf("\n arr[low]=%f",arr[low]); printf("\t arr[high]=%f",arr[high]);
    for (;;) {
    if (high <= low) /* One element only */
    return arr[median] ;
    if (high == low + 1) { /* Two elements only */
    if (arr[low] > arr[high])
    ELEM_SWAP(arr[low], arr[high]) ;
    return arr[median] ;
    }
//     printf("\n low=%d",low); printf("\t high=%d",high);
//         printf("\n arr0[low]=%f",arr[low]); printf("\t arr0[high]=%f",arr[high]);
    /* Find median of low, middle and high items; swap into position low */
    middle = (low + high) / 2;
    if (arr[middle] > arr[high])
    ELEM_SWAP(arr[middle], arr[high]) ;
    if (arr[low] > arr[high])
    ELEM_SWAP(arr[low], arr[high]) ;
    if (arr[middle] > arr[low])
    ELEM_SWAP(arr[middle], arr[low]) ;
    /* Swap low item (now in position middle) into position (low+1) */
    ELEM_SWAP(arr[middle], arr[low+1]) ;
    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;) {
    do ll++; while (arr[low] > arr[ll]) ;
    do hh--; while (arr[hh] > arr[low]) ;
    if (hh < ll)
    break;
    ELEM_SWAP(arr[ll], arr[hh]) ;
    }
    /* Swap middle item (in position low) back into correct position */
//         printf("\n low=%d",low); printf("\t high=%d",high);
//    printf("\n arr1[0]=%f",arr[0]); printf("\t arr1[1]=%f",arr[1]); printf("\t arr1[2]=%f",arr[2]);
//    printf("\n arr1[3]=%f",arr[3]); printf("\t arr1[4]=%f",arr[4]); printf("\t arr1[5]=%f",arr[5]);
//    printf("\n arr1[6]=%f",arr[6]); printf("\t arr1[7]=%f",arr[7]); printf("\t arr1[8]=%f",arr[8]);
//    printf("\t arr1[9]=%f",arr[9]);
        ELEM_SWAP(arr[low], arr[hh]) ;
//         printf("\n low=%d",low); printf("\t high=%d",high);
//    printf("\n arr2[0]=%f",arr[0]); printf("\t arr2[1]=%f",arr[1]); printf("\t arr2[2]=%f",arr[2]);
//    printf("\n arr2[3]=%f",arr[3]); printf("\t arr2[4]=%f",arr[4]); printf("\t arr2[5]=%f",arr[5]);
//    printf("\n arr2[6]=%f",arr[6]); printf("\t arr2[7]=%f",arr[7]); printf("\t arr2[8]=%f",arr[8]);
//    printf("\t arr2[9]=%f",arr[9]);
    /* Re-set active partition */
    if (hh <= median)
    low = ll;
    if (hh >= median)
    high = hh - 1;
    }
//     printf("\n arr[low]=%f",arr[low]); printf("\n arr[high]=%f",arr[high]);
    return arr[median] ;
}





/////////////////////////////////////////////////////////////////////////

int yprime(
		   double	yp[],
 		   double	y[],
              int   kk,
              int   noyau
		   )
{
    double	r1,r2;

    r1 = sqrt((y[0]+mu)*(y[0]+mu) + y[2]*y[2]); 
    r2 = sqrt((y[0]-mus)*(y[0]-mus) + y[2]*y[2]);

    /* Print warning if dividing by zero. */    
    if (r1 == 0.0 || r2 == 0.0 ){
        //mexWarnMsgIdAndTxt( "MATLAB:yprime:divideByZero", 
        //        "Division by zero!\n");
    }
     
//      aa=malloc (4 * sizeof(elem_type));
//           aa=1;
//           aa[1]=y[1];
//      aa[2]=y[2];aa[3]=y[3];
     
//                    printf("\n input aa[0]=%f",aa);
//      printf("\t input aa[1]=%f",aa[1]);
//              printf("\t input aa[2]=%f",aa[2]);
//      printf("\t input aa[3]=%f",aa[3]);
     
      yp[noyau]=quick_select_median(y, kk);
     
//              printf("\n input y[0]=%f",y[0]);
//      printf("\t input y[1]=%f",y[1]);
//              printf("\t input y[2]=%f",y[2]);
//      printf("\t input y[3]=%f",y[3]);
     
//      yp=mexMedian(y);
    
//     yp[0] = y[1];
//     yp[1] = 2*y[3]+y[0]-mus*(y[0]+mu)/(r1*r1*r1)-mu*(y[0]-mus)/(r2*r2*r2);
//     yp[2] = y[3];
//     yp[3] = -2*y[1] + y[2] - mus*y[2]/(r1*r1*r1) - mu*y[2]/(r2*r2*r2);
    return 1;
}


int fastmedianRFI(const double *spec, size_t sizeRow, size_t sizeCol, size_t sizeWidthFilter, const char *flag, double *out) {
    double *y; 
    size_t m,n; 
    int kk,noyau,Min,Max,Val,ct;
    double *aa;
    
    int i,k,l;
    int nbt,nbf;

	nbt=sizeCol;
    nbf=sizeRow;
	
	ct=0;
	int widthFilter = (int) sizeWidthFilter;
    for (i=0;i<nbt;i++){
        for (k=0;k<widthFilter;k++){
            noyau=i*nbf+k;
            out[noyau]=spec[noyau];
        }
        
        for (k=nbf-widthFilter;k<nbf;k++){
            noyau=i*nbf+k;
            out[noyau]=spec[noyau];
        }
    }
    kk=2*widthFilter;
    aa=malloc (kk * sizeof(double));
    for (i=0;i<nbt;i++){
        for (k=0;k<nbf;k++){
            noyau=i*nbf+k;
            
            if (flag[noyau]==1){
				Min=i*nbf+MAX((k-widthFilter), 0);
				Max=i*nbf+MIN((k+widthFilter), nbf - 1);
            
				for (l=0;l<2*widthFilter;l++){
					aa[l] = 0;
				}
			
				Val=0;
				for (l=Min;l<Max;l++){
					if (flag[l] != 1){
						aa[Val] = spec[l];
						Val++;
                    }
                }
				kk=Max-Min;
				yprime(out,aa,kk,noyau);
            }
            else{
                out[noyau]=spec[noyau];
            }

        } 
    }
    printf("\n ct=%d",ct);
    free(aa);
    return 1;
}