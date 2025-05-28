import cupy as cp

if cp.cuda.is_available():
    print(f"CuPy is available. CUDA Runtime Version: {cp.cuda.runtime.runtimeGetVersion()}")
    print(f"CuPy Driver Version: {cp.cuda.runtime.driverGetVersion()}")
else:
    print("CUDA is not available according to CuPy.")