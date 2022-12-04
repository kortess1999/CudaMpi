#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#define ARR_SIZE 20
#define PORTION_SIZE 25
#define INPUT_DATA 30
#define OUTPUT_DATA 35

void addWithCuda(int* data, const int arraySize, double* output, const int outputSize, const int blockSize);

int getFileSize(const char* file_name) {

    int _file_size = 0;
    FILE* fd;
    fd = fopen(file_name, "rb");
    if (fd == NULL) {
        _file_size = -1;
    }
    else {
        fseek(fd, 0, SEEK_END);
        _file_size = ftell(fd);
        fclose(fd);
    }
    return _file_size;
}

int main()
{
    MPI_Init(NULL, NULL);

    MPI_Status status;
    int threadNum, threadsCount;

    MPI_Comm_rank(MPI_COMM_WORLD, &threadNum);
    MPI_Comm_size(MPI_COMM_WORLD, &threadsCount);

    if (threadNum == 0) {
        FILE* fileIn;
        FILE* fileOut;
        char inputFile[1024];
        char outputFile[1024];
        int data;
        int arraySize, stepCount = 0;
        float summa = 0;
        int startTime, endTime;
        long fileLen;
        double num;

        printf("Input fileName\n");
        scanf("%s", inputFile);
        fileIn = fopen(inputFile, "rb");

        if (!fileIn) {
            perror("[Missing file] ");
            return 0;
        }

        fileLen = getFileSize(inputFile);
        printf("FileSize: %d\n", fileLen);

        printf("Output fileName\n");
        scanf("%s", outputFile);
        fileOut = fopen(outputFile, "w");

        printf("Input N:\n");
        scanf("%d", &arraySize);

        int numbersCount = fileLen / sizeof(int);
        const int blockCount = ceil(numbersCount / arraySize);
        int* inputData = (int*)malloc(fileLen);
        double* outputData = (double*)malloc((numbersCount / arraySize + 1) * sizeof(double));

        if (inputData == NULL || outputData == NULL) {
            printf("Ошибка выделения памяти");
        }
        int i = 0;
        while (fread(&data, sizeof(int), 1, fileIn)) {
            inputData[i++] = data;
        }

        //Communication
        //1 portion for root 1 for GPU for now
        int portionSize = numbersCount / arraySize / threadsCount;

        for (int i = 1; i < threadsCount; i++) {
            MPI_Send(&arraySize, 1, MPI_INT, i, ARR_SIZE, MPI_COMM_WORLD);
            MPI_Send(&portionSize, 1, MPI_INT, i, PORTION_SIZE, MPI_COMM_WORLD);
            for (int j = 0; j < portionSize * arraySize; j++) {
                MPI_Send(&inputData[i * portionSize * arraySize + j], 1, MPI_INT, i, INPUT_DATA, MPI_COMM_WORLD);
            }
        }
        startTime = clock();
        // root calculations
        for (int j = 0; j < portionSize; j++) {
            summa = 0;
            for (int i = 0; i < arraySize; i++) {
                summa += inputData[j * arraySize + i];
            }
            outputData[j] = (summa / arraySize);
        }

        // Last data calculation
        summa = 0;
        for (int i = (numbersCount / arraySize / threadsCount) * threadsCount; i < numbersCount; i++)
            summa += inputData[i];

        outputData[i / arraySize] = (summa / arraySize);

        for (int i = 1; i < threadsCount; i++) {
            for (int j = 0; j < portionSize + 1; j++) {
                MPI_Recv(&num, 1, MPI_DOUBLE, i, OUTPUT_DATA, MPI_COMM_WORLD, &status);
                outputData[portionSize * i + j] = num + 0;
            }
        }

        endTime = clock();
        // Feeding result

        for (int i = 0; i < blockCount; i++) {
            fprintf(fileOut, "%f\n", outputData[i]);
        }
        printf("TIME: %f BYTES_SIZE: %d\n", (float)(((float)endTime - (float)startTime) / (float)CLOCKS_PER_SEC), fileLen);

        free(inputData);
        free(outputData);

        fclose(fileIn);
        fclose(fileOut);
    }
    else if (threadNum == 1) {
        int arrayBlockSize, portionSize, num;
        MPI_Recv(&arrayBlockSize, 1, MPI_INT, 0, ARR_SIZE, MPI_COMM_WORLD, &status);
        MPI_Recv(&portionSize, 1, MPI_INT, 0, PORTION_SIZE, MPI_COMM_WORLD, &status);

        int* input = (int*)malloc(portionSize * arrayBlockSize * sizeof(int));
        double* output = (double*)malloc((portionSize + 1) * sizeof(double));

        for (int i = 0; i < portionSize * arrayBlockSize; i++) {
            MPI_Recv(&num, 1, MPI_INT, 0, INPUT_DATA, MPI_COMM_WORLD, &status);
            input[i] = num + 0;
        }

        addWithCuda(input, portionSize* arrayBlockSize, output, portionSize, arrayBlockSize);

        for (int i = 0; i < portionSize + 1; i++) {
            MPI_Send(&output[i], 1, MPI_DOUBLE, 0, OUTPUT_DATA, MPI_COMM_WORLD);
        }
    }
    return 0;
}
