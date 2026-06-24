using System;
using BenchmarkDotNet.Attributes;
using FastMatrixEngine;

namespace MathTests
{
    // [MemoryDiagnoser] tracks memory allocation and garbage collection in your final table.
    // [RankColumn] adds a nice ranking (1, 2) to the output table.
    [MemoryDiagnoser]
    [RankColumn]
    public class MatrixBenchmark
    {
        // 1000x1000 is computationally heavy: 1,000,000,000 operations (N^3)
        private const int N = 1000; 
        
        private float[,] _matrixA;
        private float[,] _matrixB;

        [GlobalSetup]
        public void Setup()
        {
            _matrixA = new float[N, N];
            _matrixB = new float[N, N];
            
            var rand = new Random(42); // Fixed seed for reproducible tests
            
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    _matrixA[i, j] = (float)rand.NextDouble();
                    _matrixB[i, j] = (float)rand.NextDouble();
                }
            }
        }

        // Baseline = true sets this as the 100% mark. Everything else is compared to it.
        [Benchmark(Baseline = true)]
        public float[,] StandardCpuMultiplication()
        {
            float[,] result = new float[N, N];
            
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    float sum = 0f;
                    for (int k = 0; k < N; k++)
                    {
                        sum += _matrixA[i, k] * _matrixB[k, j];
                    }
                    result[i, j] = sum;
                }
            }
            
            return result;
        }

        [Benchmark]
        public float[,] GpuMultiplication()
        {
            // Call your new high-performance library!
            return GpuMatrixMultiplier.Multiply(_matrixA, _matrixB);
        }
    }
}