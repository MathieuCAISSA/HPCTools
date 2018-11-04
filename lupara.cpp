#include <iostream>
#include <fstream>
#include <time.h>
#include <stdlib.h>
#include <omp.h>

using namespace std;

// MATRICE OPERATIONS ---------------------------------------------------------------------------------------------------------------------------------------------------
// MATRICE INITIALIZATION -------------------------------------
double** zero(int n){
	double** res = new double*[n];
	int i,j;
	for (i = 0; i < n; ++i){
		res[i]= new double[n];
		for (j = 0; j < n; ++j){
			res[i][j] = 0;
		}
	}
	return res;
}

double** eye(int n){
	double** res = zero(n);
	int i;
	for (i = 0; i < n; ++i){
		res[i][i]=1;
	}
	return res;
}

double** randM(int n){
	double** res = zero(n);
	int i,j;
	for (i = 0; i < n; ++i){
		for (j = 0; j < n; ++j){
			double randd = (double)rand() / RAND_MAX;
			res[i][j] = randd;
		}
	}
	return res;
}

// OPERATIONS -----------------------------------------------

void mmx(double* v, double* y, double x, int n){
	int i;
	for (i = 0; i < n; ++i){
		v[i]=v[i]-y[i]*x;
	}
}


// SOLVER ---------------------------------------------------------------------------------------------------------------------------------------------------------------

void down(double** L, double** b, int n){
	double val;
	int i,j,k;
	#pragma omp parallel for private(i,j,k) reduction(+:val)
	for (i = 0; i < n; ++i){
		for (j = 0; j < n; ++j){
			val = 0.0;
			for (k = 0; k < j; ++k){
				val+=L[j][k]*b[k][i];
			}
			b[j][i]=b[j][i]-val;
		}
	}
}

void up(double** U, double** b, int n){
	double val;
	int i,j,k;
	#pragma omp parallel for private(i,j,k) reduction(+:val)
	for (i = 0; i < n; ++i){
		for (j = n-1; j >=0; --j){
			val = 0.0;
			for (k = j+1; k < n; ++k){
				val+=U[j][k]*b[k][i];
			}
			b[j][i]=(b[j][i]-val)/U[j][j];
		}
	}
}


//LU WITHOUT PIVOT ------------------------------------------------------------------------------------------------------------------------------------------------------

double** luc(double** A, double** B, int n){
	double** L = eye(n);
	int i,k,j;
	for (i = 0; i < n; ++i){
		for (k = i+1; k < n; ++k){
			L[k][i]=A[k][i]/A[i][i];
		}
		for (j = i+1; j < n; ++j){
			mmx(A[j],A[i], L[j][i], n);
		}
	}
	down(L,B,n);
	up(A,B,n);
	return B;
}


// MAIN -----------------------------------------------------------------------------------------------------------------------------------------------------------------

int main(int argi, char *argc[]) {
	int taille = atoi(argc[1]);
	double** y = randM(taille);
	double** m = randM(taille);

	double tt1 = omp_get_wtime();
	double** res = luc(m, y, taille);
	double tt2 = omp_get_wtime();
	printf("time = %fs \n",tt2-tt1);
	return 0;
}

//icc -xcore-avx2 -fopenmp -ipo -Ofast -o lupara lupara.cpp
//./lupara 1000