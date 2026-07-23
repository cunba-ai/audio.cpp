"""
test_capi_devices.py — 验证设备枚举 API
"""
import ctypes
import os
import sys

DLL = os.path.join("build", "windows-cpu-release", "bin", "audiocpp.dll")
lib = ctypes.CDLL(DLL)

class DeviceInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 128),
        ("description", ctypes.c_char * 256),
        ("backend", ctypes.c_int),
        ("device_id", ctypes.c_int),
        ("type", ctypes.c_int),
        ("memory_total", ctypes.c_uint64),
        ("memory_free", ctypes.c_uint64),
    ]

lib.audiocpp_version.restype = ctypes.c_char_p
lib.audiocpp_device_count.restype = ctypes.c_int
lib.audiocpp_device_info.restype = ctypes.c_int
lib.audiocpp_device_info.argtypes = [ctypes.c_int, ctypes.POINTER(DeviceInfo)]
lib.audiocpp_list_devices.restype = None

print(f"audiocpp version: {lib.audiocpp_version().decode()}")
count = lib.audiocpp_device_count()
print(f"device count: {count}\n")

backend_names = {0: "CPU", 1: "CUDA", 2: "Vulkan", 3: "Metal", 4: "SYCL", 5: "BEST"}
type_names = {0: "CPU", 1: "GPU", 2: "IGPU"}

for i in range(count):
    info = DeviceInfo()
    ret = lib.audiocpp_device_info(i, ctypes.byref(info))
    if ret == 0:
        print(f"  [{i}] backend={backend_names.get(info.backend, '?'):<8} "
              f"type={type_names.get(info.type, '?'):<5} "
              f"device_id={info.device_id} "
              f"mem={info.memory_free / 1e9:.1f}/{info.memory_total / 1e9:.1f} GB "
              f"name={info.name.decode('utf-8', 'replace')}")
    else:
        print(f"  [{i}] ERROR: device_info returned {ret}")

print("\n=== audiocpp_list_devices() output ===")
lib.audiocpp_list_devices()
