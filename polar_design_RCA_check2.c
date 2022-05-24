// Polar Code Design Program Based on RCA
// (C) Hideki Ochiai
//
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( int argc, char *argv[] ){ 
  FILE *WDATA1, *WDATA2, *WDATA3, *WDATA4;
  char *filename;

  double R, EsN0, EsN0dB, R_true;
  int repeat, nn, MODE, K, N;
  double BLER;  
  
  if( argc < 3 ){
    printf("Usage: %s -n (N = 2^n) -desEsN0dB \n", argv[0] );

    int ii;
    for( ii = 1; ii <= 20; ii ++ ){
      printf("n = %d: N = %d \t",  ii, (1 << ii ) );
      if( ii % 3 == 0 )
	printf("\n");
    }
    printf("\n");
    exit(0);
  }
  int OBJ;
  nn = atoi( argv[1] );

  double desEsN0dB, desEsN0;
  desEsN0dB = atof( argv[2] );

  N = ( 1 << nn );

  int *FZN;
  FZN = (int *)malloc(sizeof(int) * N );

  filename = ( char *)malloc( 1000 * sizeof( char ) );

  double *SNRset, dummy, *SNRdB_in;
  double get_SNR_checkRCA( int N, double *SNRdB_in, double *SNRset );

  SNRset = (double *)malloc(sizeof(double) * N );
  SNRdB_in = (double *)malloc(sizeof(double) * N );
    
  int i;

  sprintf( filename, "SNRdB_N%d_desEsN0%1.3f.dat", N, desEsN0dB );
  WDATA1 = fopen(filename, "w");

  for( i = 0; i < N; i ++ ){
    SNRdB_in[i] = EsN0dB;
  }
  dummy = get_SNR_checkRCA( N, SNRdB_in, SNRset );

  for( i = 0; i < N; i ++ ){
    fprintf( WDATA1, "%d, %e\n", i, SNRset[i] );
  } 

  fclose( WDATA1 );

  return 0;
}



double get_SNR_checkRCA( int N, double *SNRdB_in, double *SNRset ){
  int i, j, k, u, L;
  double SimpleRCA( double xi );

  double *z, R, *no_one_z;
  
  z = (double *)malloc(sizeof(double) * N );
  if( z == NULL ){
    printf("Memory cannot be allocated\n");
    exit(0);
  }  

  L = (int) log2( N );
  
  long double ln_S, SNRdB;
  long double TT, ln_Stmp, ln_Snew, ln2, ltmp;
  ln2 = logl( 2.0 );

  for( i = 0; i < N; i ++ ){
    z[i] = 0.1 * SNRdB_in[i] / log10( M_E ); 
  }
  double xi0, xi1, L0, L1;
  int J;

  double max2( double a, double b );
  
  for( i = 0; i < L; i ++ ){
    J = ( 1 << (i+1) );

    for( k = 0; k < N/J; k ++ ){

      for( j = 0; j < J / 2; j ++ ){
	
	xi0 = z[k * J + j ];
	xi1 = z[k * J + j + J/2 ];
	L0 = SimpleRCA( xi0 );
	L1 = SimpleRCA( xi1 );
	
	z[k * J + j] = SimpleRCA( max2( L0, L1 )  + log( 1.0 + exp( - fabs( L0 - L1 ) ) ) );
	z[k * J + j + J/2] = max2( xi0, xi1 ) + log( 1.0 + exp( - fabs( xi0 - xi1 ) ) );
      }
    }
  }

  for( i = 0; i < N; i ++ ){
    SNRset[i] = 10.0 * log10( M_E ) * z[i];
  }

  return 1.0;
}

double max2( double a, double b)
{
  if( a > b ) return a;
  else return b;
}


double SimpleRCA( double xi )
{
  double alpha =  1.16125;
  double h21 = 1.396634, h22 = 0.872764, h23 = 1.148562;
  double h31 = 1.266967, h32 = 0.938175, h33 = 0.986830;
  double B, g, L, A, rt;
  
  if( xi < -11.3143 ){
    B = log( 2.0 ) + 2.0 * log( log( 2.0 ) )+ 2.0 * log( alpha )- 2.0 * xi;
    rt = log( B + ( 1.0 / B - 1.0 ) * log( B ) ) - log( 2.0 );
    return rt;
  }
  g = exp( xi );
  if( g > 10.0 ){  
    return log( log( 2.0 ) ) + log( alpha ) - g - 0.5 * xi;
  } else if ( g < 0.04 ){
    L = 1.0 - ( g - pow( g, 2.0 ) + 4.0 / 3.0 * pow( g, 3.0 ) )/ log( 2.0 );
  } else if( g < 1.0 ){
    L = 1.0 - pow( 1.0 - exp( - h21 * pow( g, h22 ) ), h23);
  } else {
    L = 1.0 - pow( 1.0 - exp( - h31 * pow( g, h32 ) ), h33);
  }
  if( L < 0.055523 ){
    A = pow( -5.0 + 24.0 * log( 2.0 ) * L + 2.0 * sqrt( 13.0 + 12.0 * log(2.0) * L * ( 12.0 * log(2.0) * L - 5.0 ) ), 1.0 / 3.0 );
    rt = log( 1.0 - 3.0 / A + A ) - 2.0 * log( 2.0 );
    return rt;
  } else if( L < 0.721452 ){
    rt = ( log( - log( 1.0 - pow( L, 1.0 / h23 ) ) ) - log( h21 ) ) / h22;
    return rt;
  } else {
    rt = ( log( - log( 1.0 - pow( L, 1.0 / h33 ) ) ) - log( h31 ) ) / h32;
    return rt;
  }

}


