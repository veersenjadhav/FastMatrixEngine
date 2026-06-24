# FastMatrixEngine
FastMatrixEngine is a high-performance, GPU-accelerated Matrix Multiplication library for .NET

Built on top of [ILGPU](https://github.com/m4rs-mt/ILGPU), this open-source package takes the heavy lifting out of N³ matrix mathematics. It completely abstracts away complex CUDA/VRAM memory management and provides a dead-simple, crash-proof C# API.

## ✨ Features

- 🏎️ **Insanely Fast:** Offloads multi-dimensional array mathematics to thousands of GPU cores.
- 🛡️ **Smart Hardware Fallback:** Automatically detects the best hardware on the user's machine:
  - `Dedicated GPU` (NVIDIA CUDA / AMD HIP) ➔ `Integrated GPU` (OpenCL) ➔ `CPU Fallback` (Ensures it never crashes!).
- 🧠 **Zero Configuration:** No need to manage VRAM pointers or write C++ kernels. Just pass in standard C# `float[,]` or `double[,]` arrays.
- 🧹 **Memory Safe:** Strict resource disposal prevents memory leaks in your GPU's VRAM.

---

## 📦 Installation

Since this is a standard .NET library, you can install it via the NuGet Package Manager:

```bash
dotnet add package FastMatrixEngine
```

## 💻 Quick Start
Using `FastMatrixEngine` is as simple as calling a single method. The library handles the CPU ➔ GPU ➔ CPU memory transfers automatically.

```C#
using FastMatrixEngine;
using System;

class Program
{
    static void Main()
    {
        // 1. Define your matrices (e.g., 1000x1000)
        float[,] matrixA = new float[1000, 1000];
        float[,] matrixB = new float[1000, 1000];

        // ... fill your matrices with data ...

        // 2. Multiply! The engine automatically detects your GPU and processes the math.
        float[,] result = GpuMatrixMultiplier.Multiply(matrixA, matrixB);

        Console.WriteLine("Matrix multiplication completed successfully on the GPU!");
    }
}
```
