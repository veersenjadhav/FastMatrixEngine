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
