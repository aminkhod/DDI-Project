# include "pch.h"

# include <stdio.h>
# include <omp.h>
# include <time.h>

# include <vector>
# include <algorithm>
# include <cstdlib>
# include <cstring>
# include <iomanip>
# include <iostream>
using namespace std;
int main()
{

	double a[200][200] = {};
	double b[200][200] = {};
	double sum = 0;
	int i, j, n, m;
	int num_proces, max_threads;
	double wtime;
	printf("enter no. raws of elements");
	scanf_s("%d", &n);
	printf("enter no. column of elements");
	scanf_s("%d", &m);
	

	for (i = 0; i < n; i++)
		for (j = 0; j < m; j++)
			if (((i + j) % 3) == 0)
				a[i][j] = 1;
				
			else
				b[i][j] = 2;
	num_proces = omp_get_num_procs();
	max_threads = omp_get_max_threads();
	wtime = omp_get_wtime();
	for (i = 0; i < n; i++)
		for (j = 0; j < m; j++)
		{

			b[i][j] = b[i][j] + a[i][j];
			//printf("\n a[%d, %d] = %d", i, j, a[i][j]);
		}
	wtime = omp_get_wtime() - wtime;
	cout << "  ----------- Sequential ------------ " << "\n";
	//cout << "  Number of processors available = " << omp_get_num_procs() << "\n";
	//cout << "  Number of threads =              " << omp_get_max_threads() << "\n";
	cout << "  Done during =                    " << wtime << "\n";
	cout << "  Sum of matrix elements is =                    " << sum << "\n";
	sum = 0;
	omp_set_num_threads(max_threads);
	wtime = omp_get_wtime();
	
	//#pragma omp parallel for reduction (+:a[:n][:m])
	//#pragma omp parallel for private(j)
	#pragma omp parallel for
	for (i = 0; i < n; i++)
	#pragma omp parallel for
		for (j = 0; j < m; j++)
		{

			b[i][j] = b[i][j] + a[i][j];
			//printf("\n a[%d, %d] = %d", i, j, a[i][j]);
		}
	wtime = omp_get_wtime() - wtime;

	cout << "\n";
	cout << "  --------- Parallel Mode ---------- " << "\n";
	cout << "  Number of processors available = " << omp_get_num_procs() << "\n";
	cout << "  Number of threads =              " << omp_get_max_threads() << "\n";
	cout << "  Done during =                    " << wtime << "\n";
	cout << "  Sum of matrix elements is =                    " << sum << "\n";


}