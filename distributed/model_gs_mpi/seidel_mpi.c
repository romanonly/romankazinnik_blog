#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include <unistd.h>

/* Решение системы методом Гаусса-Зейделя */

// error must decrease within counter_tolerate
const int max_counter = 1000, counter_tolerate = 3;
//local solver GS iterations
const int local_solver_num = 10;

double analytc_err[max_counter];

double *x;
int ProcNum; //Количество процессов
int ProcRank; //Ранг процесса

void Process(int, double,double,int);

int main(int agrc, char* argv[])
{
    double t1;
    int n, i, is_sync;
	double eps;
	FILE *Out;

	//Инициализация MPI
	MPI_Init(&agrc, &argv);
    t1 = MPI_Wtime();

	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	//Это все выполняем в 0 процессе
	if (ProcRank == 0) {
		printf("Gauss-Seidel method\n\n ProcNum=%d ProcRank=%d", ProcNum, ProcRank);
		printf("n = ");
		if (scanf("%d", &n) > 0) printf("OK\n");
		printf("eps = ");
		if (scanf("%lf", &eps) > 0) printf("OK\n");
        printf("1/0 (sync or async)= ");
        if (scanf("%d", &is_sync) > 0) printf("OK\n");

		printf("\n");
	}
	//Рассылаем полученные значения всем процессам
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&is_sync, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	x = (double*)malloc((n+1)*sizeof(double));

	//Начальное приближение
	for (i = 0; i <= n; i++)
		x[i] = 0;

	Process(n, eps, t1, is_sync);

	//Запись в файл
	if (ProcRank == 0) {
		Out = fopen("output.txt", "w");
		for (i = 0; i <= n; ++i) {
			fprintf(Out, "%lf\n", x[i]);
		}
		fclose(Out);
	
		printf("\nResult written in file output.txt\n");
	}

	//Завершение MPI
	MPI_Finalize();

	return(0);
}

void mpi_process_sync(double eps, double t1, int n, double *prev_x, double **C, double **A, double * f);
void mpi_process_async(double eps, double t1, int n, double *prev_x, double *curr_x, double **C, double **A, double * f);

void Process(int n, double eps, double t1, int is_sync)
{
    int i, j;
    double *a, *b, *c, *f, *p;
    double *prev_x, *curr_x;
    double **A, **C;
    
    // Create matrix system

    a = (double*)malloc((n+1)*sizeof(double));
    b = (double*)malloc((n+1)*sizeof(double));
    c = (double*)malloc((n+1)*sizeof(double));
    f = (double*)malloc((n+1)*sizeof(double));
    p = (double*)malloc((n+1)*sizeof(double));

    prev_x = (double*)malloc((n+1)*sizeof(double));
    curr_x = (double*)malloc((n+1)*sizeof(double));

    A = (double**)malloc((n+1)*sizeof(double*));
    for (i = 0; i <= n; ++i)
        A[i] = (double*)malloc((n+1)*sizeof(double));

    C = (double**)malloc((n+1)*sizeof(double*));
    for (i = 0; i <= n; ++i)
        C[i] = (double*)malloc((n+1)*sizeof(double));

    //Заполняем столбцы данных
    b[0] = 1.;
    c[0] = 0.;
    f[0] = 1.;
    p[0] = 1.;

    for (i = 1; i < n; i++) {
        a[i] = 1.;
        b[i] = -2.;
        c[i] = 1.;
        f[i] = 2./(i*i + 1);
        p[i] = 2.;
    }

    f[n] = -n/3.;
    p[n] = 1.;

    //Формируем матрицу коэффициентов
    for (i = 0; i <= n; i++)
    for (j = 0; j <= n; j++)
        A[i][j] = 0.;

    A[0][0] = b[0]; A[0][1] = c[0];

    for (i = 1; i < n; i++) {
        A[i][i] = b[i];
        A[i][i+1] = c[i];
        A[i][i-1] = a[i];
    }

    for (j = 0; j <= n; j++)
        A[n][j] = p[j];

    //Формируем расчетную матрицу
    for (i = 0; i <= n; i++)
    for (j = 0; j <= n; j++) {
        if (i == j) {
            C[i][j] = 0;
        }
        else {
            C[i][j] = -A[i][j]/A[i][i];
        }
    }

    // run solver 
    if (is_sync)
        return mpi_process_sync(eps, t1, n, prev_x, C, A, f);
    else
        return mpi_process_async(eps, t1, n, prev_x, curr_x, C, A, f);
}
        
int print_mpi_log(double eps, double t1, int counter, double *prev_x, double *x, int n, double **A, double *f);

void mpi_process_sync(double eps, double t1, int n, double *prev_x, double **C, double **A, double * f)
{
    int counter = 0;
    int i, j;
    double ProcSum, TotalSum;
    

    //На каждом процессе будет вычисляться частичная сумма длиной k
    int k = (n+1) / ProcNum;
    //То есть будут суммироваться только те элементы, которые лежат между i1 и i2
    int i1 = k * ProcRank;
    int i2 = k * (ProcRank + 1);
    if (ProcRank == ProcNum - 1) i2 = n+1;

    printf("Sync Calculating started: %d \n\n", ProcRank);

    //Собственно расчет
    while(1) {
        for (i = 0; i <= n; i++)
            prev_x[i] = x[i];

        for (i = 0; i <= n; i++) {
            ProcSum = 0.0;

            for (j = 0; j < i; j++)
                if ((i1 <= j) && (j < i2))
                     ProcSum += C[i][j]*x[j];

            for (j = i; j <= n; j++)
                if ((i1 <= j) && (j < i2))
                     ProcSum += C[i][j]*prev_x[j];

            TotalSum = 0.0;

            //Сборка частичных сумм ProcSum на процессе с рангом 0

            // All reduce from all nodes into node 0
            MPI_Reduce(&ProcSum, &TotalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            // Update all nodes from node 0             
            MPI_Bcast(&TotalSum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            
            x[i] = TotalSum + f[i]/A[i][i];
        }

        counter++;

        if (-1==print_mpi_log(eps, t1, counter, prev_x, x, n, A, f))
            return;
    } // end while
}

void mpi_process_async(double eps, double t1, int n, double *prev_x, double *curr_x, double **C, double **A, double * f)
{
    double eps1;
    int counter = 0;
    int i, j, ii;
    double ProcSum, TotalSum;
    int tag=1;

	//На каждом процессе будет вычисляться частичная сумма длиной k
	int k = (n+1) / ProcNum;
	//То есть будут суммироваться только те элементы, которые лежат между i1 и i2
	int i1 = k * ProcRank;
	int i2 = k * (ProcRank + 1);
	if (ProcRank == ProcNum - 1) i2 = n+1;


    // take from previous
    int ProcRankPrev = ProcRank-1;
    if (ProcRankPrev < 0) ProcRankPrev = ProcNum-1;
    int i1_prev = k * ProcRankPrev;
    int i2_prev = k * (ProcRankPrev + 1);
    if (ProcRankPrev == ProcNum - 1) i2_prev = n+1;

    // send to next
    int ProcRankNext = ProcRank+1;
    if (ProcRankNext == ProcNum) ProcRankNext = 0;


    printf("Async Calculating started: %d \n\n", ProcRank);

    //Собственно расчет

    for (i = 0; i <= n; i++)
       curr_x[i] = x[i];


    // Update only i1 .. i2 unknowns and ring-reduce
    while(1) {

        MPI_Status status;
        for (i = 0; i <= n; i++)
           prev_x[i] = curr_x[i];

        // inner loop for local convergence
       for (ii=0; ii<local_solver_num; ii++) {

            eps1 = 0.; // inner solver convergence epsilon

            for (i = i1; i < i2; i++) {
                ProcSum = 0.0;

                for (j = 0; j < i; j++)
                       ProcSum += C[i][j]*curr_x[j];

                for (j = i; j <= n; j++)
                         ProcSum += C[i][j]*prev_x[j];

                curr_x[i] = ProcSum + f[i]/A[i][i];      

                eps1 += fabs(curr_x[i] - prev_x[i]);
            }
            if (eps1 < eps) 
                break;
            else {
                for (i = i1; i < i2; i++)
                   prev_x[i] = curr_x[i];
            }
        }
        if ((ProcRank == 0) && (counter % 10 == 0))
            ;//printf("\n inner solver eps=%f with %d-th iteration\n", eps1, ii);

        // Compute current Process
        // Send to next Process
        // take from Previous Process

        // Process 0 can not be blocked initially
        if (ProcRank != 0) 
            MPI_Recv (&(curr_x[i1_prev]),i2_prev - i1_prev,MPI_INT, ProcRankPrev,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Send(&(curr_x[i1]),i2 - i1,MPI_INT,ProcRankNext,tag,MPI_COMM_WORLD);
        // Now process 0 can receive from the last process.
        if (ProcRank == 0) 
            MPI_Recv (&(curr_x[i1_prev]),i2_prev - i1_prev,MPI_INT, ProcRankPrev,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        
        counter++;

        if (-1==print_mpi_log(eps, t1, counter, prev_x, curr_x, n, A, f)) {           
            if (ProcRank == 0) // send local solution to global x
                for (i = 0; i <= n; i++)
                   x[i] = curr_x[i];                
            return;
        }
    } // end while
}

int print_mpi_log(double eps, double t1, int counter, double *prev_x, double *x, int n, double **A, double *f)
{
    double norm=0;
    int i, j;    
    double t2;
        //Считаем невязку
	if (ProcRank == 0) {
        	norm = 0;
        	for (i = 0; i <= n; i++)
        	    norm += (x[i] - prev_x[i])*(x[i] - prev_x[i]);
        	norm = sqrt(norm);

            // abs(AX - f)
            double err_axf = 0, norm_f = 0., sum_ax;
            for (i = 0; i <= n; i++) {
                sum_ax = 0.;
                for (j = 0; j <= n; j++) 
                    sum_ax += A[i][j]*x[j];
                err_axf += fabs(sum_ax-f[i]);
                norm_f += fabs(f[i]);
            }
            err_axf /= norm_f;

	        if (counter % 10 == 0) {
               //sleep(1);
               t2 = MPI_Wtime();
               // printf("counter=%2d.   norm=%lf analytical error=%lf %1.2f(seconds) \n", counter, norm, err_axf, t2-t1);fflush(stdout);
               printf("counter=%2d.   analytical error=%lf %1.2f (seconds) \n", counter, err_axf, t2-t1);fflush(stdout);
            }

            norm = fmax(norm, err_axf);
            analytc_err[counter] = err_axf;
            // quit when error increases
            if (counter > counter_tolerate && analytc_err[counter]>analytc_err[counter-counter_tolerate])
                norm = 0.;
	}

    MPI_Bcast(&norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (norm < eps || counter > max_counter) {
        if (ProcRank == 0) 
            printf("counyter=%2d.   norm=%lf analytical error=%lf \n", counter, norm, analytc_err[counter]);
        else //if (ProcRank == 0) 
            printf(" exit process ProcRank=%2d.\n", ProcRank);        
        return -1;
    }
    return 1;
}
