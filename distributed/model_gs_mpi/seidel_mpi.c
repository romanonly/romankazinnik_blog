#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include <unistd.h>

/* Решение системы методом Гаусса-Зейделя */

double *x;
int ProcNum; //Количество процессов
int ProcRank; //Ранг процесса
void Process(int, double,double);
int main(int agrc, char* argv[])
{
    double t1;
    	int n, i;
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
		printf("\n");
	}
	//Рассылаем полученные значения всем процессам
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	x = (double*)malloc((n+1)*sizeof(double));

	//Начальное приближение
	for (i = 0; i <= n; i++)
		x[i] = 0;

	Process(n, eps, t1);

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

void Process(int n, double eps, double t1)
{
    int i, j;
    double *a, *b, *c, *f, *p;
    double *prev_x, *curr_x;
    double **A, **C;
    


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

    if (1)
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

    if (ProcRank == 0) printf("Calculating started\n\n");

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
            MPI_Reduce(&ProcSum, &TotalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
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
    int counter = 0;
    int i, j;
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


    if (ProcRank == 0) printf("Calculating started\n\n");

    //Собственно расчет
        for (i = 0; i <= n; i++)
           curr_x[i] = x[i];

    while(1) {

        for (i = 0; i <= n; i++)
           prev_x[i] = curr_x[i];


        //for (i = 0; i <= n; i++) {
        for (i = i1; i < i2; i++) {
            ProcSum = 0.0;

            for (j = 0; j < i; j++)
                     ProcSum += C[i][j]*curr_x[j];

            for (j = i; j <= n; j++)
                     ProcSum += C[i][j]*prev_x[j];

            curr_x[i] = ProcSum + f[i]/A[i][i];            
        }


        MPI_Status status;
        // Send to next Process: k+1
        // take from Previous Process: k-1
        // Compute current Process: k
        if (1) {
            //    int token;
            int world_rank = ProcRank;
            //int world_size = ProcNum;
            if (world_rank != 0) {
                //MPI_Recv(&token, 1, MPI_INT, world_rank - 1, 0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv (&(curr_x[i1_prev]),i2_prev - i1_prev,MPI_INT, ProcRankPrev,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);//&status);
                //printf("Process %d received token %f from process %d\n", world_rank, curr_x[i1_prev], world_rank - 1);
            } 
            //MPI_Send(&token, 1, MPI_INT, (world_rank + 1) % world_size,0, MPI_COMM_WORLD);
            MPI_Send(&(curr_x[i1]),i2 - i1,MPI_INT,ProcRankNext,tag,MPI_COMM_WORLD);

            // Now process 0 can receive from the last process.
            if (world_rank == 0) {
                //MPI_Recv(&token, 1, MPI_INT, world_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv (&(curr_x[i1_prev]),i2_prev - i1_prev,MPI_INT, ProcRankPrev,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);//&status);
                //printf("Process %d received token %f from process %d\n", world_rank, curr_x[i1_prev], world_size - 1);
            }
        }
        

        counter++;

        if (-1==print_mpi_log(eps, t1, counter, prev_x, curr_x, n, A, f))
            return;
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
            double err_axf = 0, sum_ax;
            for (i = 0; i <= n; i++) {
                sum_ax = 0.;
                for (j = 0; j <= n; j++) 
                    sum_ax += A[i][j]*prev_x[j];
                err_axf += fabs(sum_ax-f[i]);
            }


	        if (counter % 10 == 0) {
               //sleep(10);
               t2 = MPI_Wtime();
        	   printf("counter=%2d.   norm=%lf analytical error=%lf %1.2f(seconds) \n", counter, norm, err_axf, t2-t1);fflush(stdout);
            }
	}
    if (ProcRank == 1) {
            if (counter % 10 == 0) {
               //sleep(5);
               t2 = MPI_Wtime();
               printf("ProcRank=%d counter=%2d. %1.2f(seconds) \n", ProcRank, counter, t2-t1);fflush(stdout);
            }        
    }

    MPI_Bcast(&norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //MPI_Request request;
    //MPI_Ibcast(&norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD,&request);

    if (norm < eps) {
        if (ProcRank == 0) 
            printf("%2d.   %lf\n", counter, norm);
        else //if (ProcRank == 0) 
            printf(" exit process ProcRank=%2d.\n", ProcRank);        
        return -1;
    }
    return 1;
}
