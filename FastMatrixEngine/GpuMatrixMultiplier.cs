using System;
using ILGPU;
using ILGPU.Runtime;

namespace FastMatrixEngine
{
    public static class GpuMatrixMultiplier
    {
        /// <summary>
        /// Multiplies two 2D float arrays using the best available hardware.
        /// </summary>
        public static float[,] Multiply(float[,] matrixA, float[,] matrixB)
        {
            int rowsA = matrixA.GetLength(0);
            int colsA = matrixA.GetLength(1); // Also the shared inner dimension
            int rowsB = matrixB.GetLength(0);
            int colsB = matrixB.GetLength(1);

            if (colsA != rowsB)
                throw new ArgumentException("Inner matrix dimensions must agree (Matrix A columns must equal Matrix B rows).");

            // --- 1. HARDWARE CHECK & INITIALIZATION ---
            // 'using' blocks ensure unmanaged hardware resources are safely disposed after use.
            using Context context = HardwareDetector.CreateContext();
            using Accelerator accelerator = HardwareDetector.GetBestAccelerator(context);

            // Load the kernel (the mini-program that runs on the GPU cores)
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, 
                ArrayView2D<float, Stride2D.DenseX>, 
                ArrayView2D<float, Stride2D.DenseX>, 
                ArrayView2D<float, Stride2D.DenseX>, 
                int>(MatrixMultiplyKernelFloat);

            // --- 2. MEMORY MANAGEMENT (GPU Allocation) ---
            // Allocate 2D blocks of memory specifically inside the GPU's VRAM.
            using var bufferA = accelerator.Allocate2DDenseX<float>(new Index2D(rowsA, colsA));
            using var bufferB = accelerator.Allocate2DDenseX<float>(new Index2D(rowsB, colsB));
            using var bufferC = accelerator.Allocate2DDenseX<float>(new Index2D(rowsA, colsB));

            // --- 3. MEMORY MANAGEMENT (Transfer CPU -> GPU) ---
            bufferA.CopyFromCPU(matrixA);
            bufferB.CopyFromCPU(matrixB);

            // --- 4. EXECUTION ---
            // Fire off the calculation. It automatically splits the workload across thousands of cores.
            kernel(bufferC.IntExtent, bufferA.View, bufferB.View, bufferC.View, colsA);
            
            // Wait for all GPU cores to finish their portion of the work
            accelerator.Synchronize();

            // --- 5. MEMORY MANAGEMENT (Transfer GPU -> CPU) ---
            // ILGPU conveniently extracts the buffer back into a standard C# 2D array.
            float[,] result = bufferC.GetAsArray2D();

            return result;
        }

        /// <summary>
        /// Overload to handle high-precision double[,] arrays. 
        /// </summary>
        public static double[,] Multiply(double[,] matrixA, double[,] matrixB)
        {
            // The exact same logic as above, but swapped for 'double' types.
            int rowsA = matrixA.GetLength(0);
            int colsA = matrixA.GetLength(1);
            int rowsB = matrixB.GetLength(0);
            int colsB = matrixB.GetLength(1);

            if (colsA != rowsB) throw new ArgumentException("Inner dimensions must agree.");

            using Context context = HardwareDetector.CreateContext();
            using Accelerator accelerator = HardwareDetector.GetBestAccelerator(context);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, int>(MatrixMultiplyKernelDouble);

            using var bufferA = accelerator.Allocate2DDenseX<double>(new Index2D(rowsA, colsA));
            using var bufferB = accelerator.Allocate2DDenseX<double>(new Index2D(rowsB, colsB));
            using var bufferC = accelerator.Allocate2DDenseX<double>(new Index2D(rowsA, colsB));

            bufferA.CopyFromCPU(matrixA);
            bufferB.CopyFromCPU(matrixB);

            kernel(bufferC.IntExtent, bufferA.View, bufferB.View, bufferC.View, colsA);
            accelerator.Synchronize();

            return bufferC.GetAsArray2D();
        }

        /// <summary>
        /// Overload to handle hilf-precision float[,] arrays. 
        /// </summary>
        public static System.Half[,] Multiply(System.Half[,] matrixA, System.Half[,] matrixB)
        {
            // The exact same logic as above, but swapped for 'double' types.
            int rowsA = matrixA.GetLength(0);
            int colsA = matrixA.GetLength(1);
            int rowsB = matrixB.GetLength(0);
            int colsB = matrixB.GetLength(1);

            if (colsA != rowsB) throw new ArgumentException("Inner dimensions must agree.");

            using Context context = HardwareDetector.CreateContext();
            using Accelerator accelerator = HardwareDetector.GetBestAccelerator(context);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView2D<System.Half, Stride2D.DenseX>, ArrayView2D<System.Half, Stride2D.DenseX>, ArrayView2D<System.Half, Stride2D.DenseX>, int>(MatrixMultiplyKernelFloatHalf);

            using var bufferA = accelerator.Allocate2DDenseX<System.Half>(new Index2D(rowsA, colsA));
            using var bufferB = accelerator.Allocate2DDenseX<System.Half>(new Index2D(rowsB, colsB));
            using var bufferC = accelerator.Allocate2DDenseX<System.Half>(new Index2D(rowsA, colsB));

            bufferA.CopyFromCPU(matrixA);
            bufferB.CopyFromCPU(matrixB);

            kernel(bufferC.IntExtent, bufferA.View, bufferB.View, bufferC.View, colsA);
            accelerator.Synchronize();

            return bufferC.GetAsArray2D();
        }

        // =====================================================================
        // ILGPU KERNELS (This code is compiled to PTX/C++ and runs on the GPU)
        // =====================================================================

        /// <summary>
        /// The GPU Kernel for Floats. 
        /// Think of this block as running on ONE single thread for ONE single pixel/cell of the output matrix.
        /// </summary>
        private static void MatrixMultiplyKernelFloat(
            Index2D index, 
            ArrayView2D<float, Stride2D.DenseX> aView, 
            ArrayView2D<float, Stride2D.DenseX> bView, 
            ArrayView2D<float, Stride2D.DenseX> cView, 
            int sharedDim)
        {
            int row = index.X;
            int col = index.Y;
            float sum = 0.0f;

            // Compute the dot product for this specific row and column
            for (int i = 0; i < sharedDim; i++)
            {
                sum += aView[new Index2D(row, i)] * bView[new Index2D(i, col)];
            }

            // Write output to Matrix C VRAM
            cView[index] = sum;
        }

        /// <summary>
        /// The GPU Kernel for Doubles.
        /// </summary>
        private static void MatrixMultiplyKernelDouble(
            Index2D index, ArrayView2D<double, Stride2D.DenseX> aView, ArrayView2D<double, Stride2D.DenseX> bView, ArrayView2D<double, Stride2D.DenseX> cView, int sharedDim)
        {
            int row = index.X;
            int col = index.Y;
            double sum = 0.0;

            for (int i = 0; i < sharedDim; i++)
            {
                sum += aView[new Index2D(row, i)] * bView[new Index2D(i, col)];
            }

            cView[index] = sum;
        }

        /// <summary>
        /// The GPU Kernel for Doubles.
        /// </summary>
        private static void MatrixMultiplyKernelFloatHalf(
            Index2D index, ArrayView2D<System.Half, Stride2D.DenseX> aView, ArrayView2D<System.Half, Stride2D.DenseX> bView, ArrayView2D<System.Half, Stride2D.DenseX> cView, int sharedDim)
        {
            int row = index.X;
            int col = index.Y;
            System.Half sum = (System.Half) 0.0f;

            for (int i = 0; i < sharedDim; i++)
            {
                sum += aView[new Index2D(row, i)] * bView[new Index2D(i, col)];
            }

            cView[index] = sum;
        }
    }
}