import nvidia_smi
import time

for x in  range(10):
    nvidia_smi.nvmlInit()

    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    print(deviceCount)
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

    nvidia_smi.nvmlShutdown()
    time.sleep(1)

# import nvidia_smi
# nvidia_smi.nvmlInit()
# deviceCount = nvidia_smi.nvmlDeviceGetCount()
# for i in range(deviceCount):
#     handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
#     util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
#     mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
#     print(f"|Device {i}| Mem Free: {mem.free/1024**2:5.2f}MB / {mem.total/1024**2:5.2f}MB | gpu-util: {util.gpu:3.1%} | gpu-mem: {util.memory:3.1%} |")