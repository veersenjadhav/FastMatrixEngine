using ILGPU;
using ILGPU.Runtime;
using System.Linq;

namespace FastMatrixEngine
{
    /// <summary>
    /// Helper class to safely detect and initialize the best available computing hardware.
    /// </summary>
    internal static class HardwareDetector
    {
        public static Context CreateContext()
        {
            // Build an ILGPU context that has access to all potential processors (CPU, GPU, etc.)
            return Context.Create(builder => builder.AllAccelerators());
        }

        public static Accelerator GetBestAccelerator(Context context)
        {
            // 1. DEDICATED GPU CHECK (NVIDIA CUDA)
            var dedicatedGpu = context.Devices.FirstOrDefault(d => 
                d.AcceleratorType == AcceleratorType.Cuda );

            if (dedicatedGpu != null)
                return dedicatedGpu.CreateAccelerator(context);

            // 2. DEDICATED GPU CHECK (AMD GPU)
            var preferredGpu = context.GetPreferredDevice(preferCPU: false);

            if (preferredGpu != null)
                return preferredGpu.CreateAccelerator(context);

            // 3. INTEGRATED GPU CHECK (OpenCL handles Intel HD Graphics, Apple M-Series, etc.)
            var integratedGpu = context.Devices.FirstOrDefault(d => 
                d.AcceleratorType == AcceleratorType.OpenCL);

            if (integratedGpu != null)
                return integratedGpu.CreateAccelerator(context);

            // 4. CPU FALLBACK (Guaranteed to be found on any computer, prevents crashes)
            var cpuFallback = context.Devices.FirstOrDefault(d => 
                d.AcceleratorType == AcceleratorType.CPU);

            return cpuFallback!.CreateAccelerator(context);
        }
    }
}