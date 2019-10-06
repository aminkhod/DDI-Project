#include "pch.h"
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <time.h>
# include <cstdlib>
# include <cstring>
# include <iomanip>

using namespace std;

int main(){
    int i,j,k,n,no,num_proces,max_threads;
    int dim[]={10,50,100,200,300,4000};
    int A[4000][4000],B[4000][4000],C[4000][4000];
    double wtime;

    num_proces=omp_get_num_procs ( );
    max_threads=omp_get_max_threads ( );

    for (i=0;i<6;i++){
        wtime = omp_get_wtime ( );
        srand(time(NULL));
        n = dim[i] * dim[i];
        for( k=0;k<dim[i];k++)
            for( j=0;j<dim[i];j++)
                A[k][j]=1+rand()% (n - 1);

        for( k=0;k<dim[i];k++)
            for( j=0;j<dim[i];j++)
                B[k][j]=1+rand()% (n - 1);

        for( k=0;k<dim[i];k++)
            for( j=0;j<dim[i];j++)
                C[k][j] = A[k][j]+B[k][j];

        wtime = omp_get_wtime ( ) - wtime;
        cout << "  ----------- Sequential ------------ " << "\n" ;
        cout << "  Number of processors available = " << omp_get_num_procs ( ) << "\n";
        cout << "  Number of threads =              " << omp_get_max_threads ( ) << "\n";
        cout << "  Dimentions of Matrix =           " << dim[i] << "\n";
        cout << "  Done during =                    " << wtime << "\n";

        for(no=1;no<=num_proces;no++){
            wtime = omp_get_wtime ( );
            omp_set_num_threads(no);
            #pragma omp parallel for private(j)
            for( k=0;k<dim[i];k++)
                for( j=0;j<dim[i];j++)
                    A[k][j]=1+rand()% (n - 1);

            #pragma omp parallel for private(j)
            for( k=0;k<dim[i];k++)
                for( j=0;j<dim[i];j++)
                    B[k][j]=1+rand()% (n - 1);

            #pragma omp parallel for private(j)
            for( k=0;k<dim[i];k++)
                for( j=0;j<dim[i];j++)
                    C[k][j]=A[k][j]+B[k][j];
            wtime = omp_get_wtime ( ) - wtime;
            cout << "\n";
            cout << "  --------- Parallel Mode ---------- " <<"\n";
            cout << "  Number of processors available = " << omp_get_num_procs ( ) << "\n";
            cout << "  Number of threads =              " << omp_get_max_threads ( ) << "\n";
            cout << "  Dimentions of Matrix =           " << dim[i] << "\n";
            cout << "  Done during =                    " << wtime << "\n";
        }
    }
}
