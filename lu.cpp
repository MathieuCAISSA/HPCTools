#include <iostream>
#include <fstream>
#include <time.h>
#include <stdlib.h>

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

void div(double* v, double x, int n){
	int i;
	for (i = 0; i < n; ++i){
		v[i]=v[i]/x;
	}
}


// SOLVER ---------------------------------------------------------------------------------------------------------------------------------------------------------------

double** solLow(double**L, double** b, int n){
	double** res = zero(n);
	double* lineI;
	int i,j;
	for (i = 0; i < n; ++i){
		lineI = b[i];
		for (j = 0; j < i; ++j){
			mmx(lineI,res[j], L[i][j], n);
		}
		res[i]=lineI;
	}
	return res;
}

double** solUP(double**L, double** b, int n){
	double** res = L;
	double* lineI;
	int i,j;
	for (i = n-1; i >=0; --i){
		lineI=b[i];
		for (j = n-1; j > i; --j){
			mmx(lineI,res[j], L[i][j], n);
		}
		div(lineI,L[i][i],n);
		res[i]=lineI;
	}
	return res;
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
	return solUP(A,solLow(L,B,n),n);
}


// MAIN -----------------------------------------------------------------------------------------------------------------------------------------------------------------

int main(int argi, char *argc[]) {

	int taille = atoi(argc[1]);
	
	double** y = randM(taille);
	double** m = randM(taille);
		
	clock_t tStartLU = clock();
	double** res = luc(m, y, taille);
	printf("Time taken for my solution with LU: %.2fs\n", (double)(clock() - tStartLU)/CLOCKS_PER_SEC);

  return 0;
}

//icc -Ofast -o lu lu.cpp
//./lu 1000