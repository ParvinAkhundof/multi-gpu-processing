slices=3
size=73257
model=1
for x in range(slices):
    start=int(size/slices*x)
    end=int(size/slices*(x+1))
    print(x )
    print(end)

# import nvidia_smi
# nvidia_smi.nvmlInit()
# deviceCount = nvidia_smi.nvmlDeviceGetCount()
# for i in range(deviceCount):
#     handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
#     util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
#     mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
#     print(f"|Device {i}| Mem Free: {mem.free/1024**2:5.2f}MB / {mem.total/1024**2:5.2f}MB | gpu-util: {util.gpu:3.1%} | gpu-mem: {util.memory:3.1%} |")