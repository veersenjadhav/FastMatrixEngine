using BenchmarkDotNet.Running;

namespace MathTests
{
    class Program
    {
        static void Main(string[] args)
        {
            // This single line triggers the entire benchmarking engine
            // and generates the beautiful text table at the end.
            var summary = BenchmarkRunner.Run<MatrixBenchmark>();
        }
    }
}