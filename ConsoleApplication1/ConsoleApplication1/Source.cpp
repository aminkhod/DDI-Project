# include "pch.h"
# include <stdio.h>
# include <omp.h>
# include <time.h>

#include <stdlib.h>
# include <vector>
# include <algorithm>
# include <cstdlib>
# include <cstring>
# include <iomanip>
# include <iostream>
using namespace std;
float main()
{

	float a[5000][5000] ;
	float sum = 0;
	int i, j, n;
	int num_proces, max_threads;
	double wtime;
	printf("enter no. raws of elements");
	scanf_s("%d", &n);
	//printf("enter no. column of elements");
	//scanf_s("%d", &m);


	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			if (((i *j) % 3) == 0)
				a[i][j] = a[i][j] = 1 + rand() % (n - 1);
			else
				a[i][j] = 0;
	num_proces = omp_get_num_procs();
	max_threads = omp_get_max_threads();
	wtime = omp_get_wtime();
	omp_set_num_threads(1);
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
		{

			sum = sum + a[i][j];
			//printf("\n a[%d, %d] = %d", i, j, a[i][j]);
		}
	wtime = omp_get_wtime() - wtime;
	cout << "  ----------- Sequential ------------ " << "\n";
	//cout << "  Number of processors available = " << omp_get_num_procs() << "\n";
	//cout << "  Number of threads =              " << omp_get_max_threads() << "\n";
	cout << "  Done during =                    " << wtime << "\n";
	cout << "  Sum of matrix elements is =                    " << (sum / (n*n)) << "\n";
	sum = 0;
	omp_set_num_threads(max_threads);
	wtime = omp_get_wtime();


	//#pragma omp for private(j)
	//#pragma omp for
//#pragma omp parallel private(i,j) reduction (+:sum)
#pragma omp parallel for private(i, j)
	for (i = 0; i < n; i++)
//#pragma omp for
//#pragma omp single
		for (j = 0; j < n; j++)
		{

			sum = sum + a[i][j];
			//printf("\n a[%d, %d] = %d", i, j, a[i][j]);
		}
	wtime = omp_get_wtime() - wtime;

	cout << "\n";
	cout << "  --------- Parallel Mode ---------- " << "\n";
	cout << "  Number of processors available = " << omp_get_num_procs() << "\n";
	cout << "  Number of threads =              " << omp_get_max_threads() << "\n";
	cout << "  Done during =                    " << wtime << "\n";
	cout << "  Sum of matrix elements is =                    " << (sum / (n*n)) << "\n";

}