"""
test_cuda_device.py — 完整的客户端使用流程演示:
1. 枚举设备 (audiocpp_device_count / device_info)
2. 找到目标 GPU
3. 用正确的 backend + device_id 加载模型
4. 运行推理

这演示了客户端(任何语言: Rust/Go/Python)该怎么调用 audio.cpp C ABI。
"""
import ctypes
import os
import sys

# ── 加载 DLL ────────────────────────────────────────────────────────────
# 客户端只需要 audiocpp.dll + audiocpp.h,不需要编译整个 audio.cpp

DLL = os.path.join("build", "windows-cuda-release", "bin", "audiocpp.dll")
if not os.path.isfile(DLL):
    DLL = os.path.join("build", "windows-cpu-release", "bin", "audiocpp.dll")

lib = ctypes.CDLL(DLL)

# ── C ABI 类型映射 ───────────────────────────────────────────────────────

class Error(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int), ("message", ctypes.c_char_p)]

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

# ── 设置函数签名 ────────────────────────────────────────────────────────

lib.audiocpp_version.restype = ctypes.c_char_p
lib.audiocpp_device_count.restype = ctypes.c_int
lib.audiocpp_device_info.restype = ctypes.c_int
lib.audiocpp_device_info.argtypes = [ctypes.c_int, ctypes.POINTER(DeviceInfo)]
lib.audiocpp_load_model.restype = ctypes.c_void_p
lib.audiocpp_load_model.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(Error),
]
lib.audiocpp_free_model.argtypes = [ctypes.c_void_p]

# ── Step 1: 枚举设备 ────────────────────────────────────────────────────

BACKEND_NAMES = {0: "CPU", 1: "CUDA", 2: "Vulkan", 3: "Metal", 4: "SYCL", 5: "BEST"}
TYPE_NAMES = {0: "CPU", 1: "GPU", 2: "IGPU"}

print(f"=== audio.cpp {lib.audiocpp_version().decode()} ===")
print(f"DLL: {DLL}\n")

count = lib.audiocpp_device_count()
print(f"发现 {count} 个设备:")
devices = []
for i in range(count):
    info = DeviceInfo()
    if lib.audiocpp_device_info(i, ctypes.byref(info)) == 0:
        devices.append(info)
        mem_str = f"{info.memory_free/1e9:.1f}/{info.memory_total/1e9} GB" if info.memory_total else "N/A"
        print(f"  [{i}] {BACKEND_NAMES.get(info.backend,'?'):<7} "
              f"{TYPE_NAMES.get(info.type,'?'):<5} "
              f"device_id={info.device_id}  "
              f"mem={mem_str}  "
              f"{info.name.decode('utf-8','replace')}")

# ── Step 2: 选择目标设备 ────────────────────────────────────────────────

# 客户端策略:优先找 CUDA GPU,没有就用 CPU
target_backend = 0  # AUDIOCPP_BACKEND_CPU
target_device = 0
target_name = "CPU"

for dev in devices:
    if dev.backend == 1:  # AUDIOCPP_BACKEND_CUDA
        target_backend = dev.backend
        target_device = dev.device_id
        target_name = dev.name.decode('utf-8', 'replace')
        break

print(f"\n选择设备: {target_name} (backend={BACKEND_NAMES[target_backend]}, device_id={target_device})")

# ── Step 3: 加载模型 ────────────────────────────────────────────────────

MODEL_PATH = os.path.abspath("models/Qwen3-ASR-0.6B")
if not os.path.isdir(MODEL_PATH):
    print(f"\n跳过模型加载测试 (模型目录不存在: {MODEL_PATH})")
    sys.exit(0)

err = Error()
print(f"\n加载模型: {MODEL_PATH}")
print(f"  backend={BACKEND_NAMES[target_backend]} device={target_device}")

model = lib.audiocpp_load_model(
    MODEL_PATH.encode("utf-8"),
    b"qwen3_asr",          # family_hint
    1,                      # AUDIOCPP_TASK_ASR
    target_backend,         # backend
    target_device,          # device_id ← 从设备枚举拿到的值
    4,                      # n_threads
    ctypes.byref(err),
)

if model:
    print(f"  ✅ 模型加载成功!")
    lib.audiocpp_free_model(model)
else:
    msg = err.message.decode("utf-8", "replace") if err.message else "(no message)"
    print(f"  ❌ 加载失败: code={err.code} msg={msg}")
