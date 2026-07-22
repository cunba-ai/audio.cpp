"""
test_capi_load.py — 决定性验证:对比 family_hint=NULL vs "qwen3_asr"
用 ctypes 动态加载 audiocpp.dll,模拟 Rust FFI 调用。
"""
import ctypes
import os
import sys

DLL = os.path.join("build", "windows-cpu-release", "bin", "audiocpp.dll")
MODEL = os.path.abspath(os.path.join("models", "Qwen3-ASR-0.6B"))

lib = ctypes.CDLL(DLL)

# --- 结构体定义 ---
class Error(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int), ("message", ctypes.c_char_p)]

# --- 函数签名 ---
lib.audiocpp_version.restype = ctypes.c_char_p
lib.audiocpp_load_model.restype = ctypes.c_void_p
lib.audiocpp_load_model.argtypes = [
    ctypes.c_char_p,  # model_path
    ctypes.c_char_p,  # family_hint
    ctypes.c_int,     # task
    ctypes.c_int,     # backend
    ctypes.c_int,     # device_id
    ctypes.c_int,     # n_threads
    ctypes.POINTER(Error),  # err
]
lib.audiocpp_free_model.argtypes = [ctypes.c_void_p]
lib.audiocpp_clear_error.argtypes = [ctypes.POINTER(Error)]

AUDIOCPP_TASK_ASR = 1
AUDIOCPP_BACKEND_CPU = 0

def try_load(label, family_hint):
    print(f"\n=== {label} (family_hint={family_hint}) ===", flush=True)
    err = Error()
    fh = family_hint.encode("utf-8") if family_hint else None
    try:
        model = lib.audiocpp_load_model(
            MODEL.encode("utf-8"), fh,
            AUDIOCPP_TASK_ASR, AUDIOCPP_BACKEND_CPU, 0, 4, ctypes.byref(err))
    except OSError as e:
        print(f"  CRASH (OSError): {e}", flush=True)
        return 1
    if model:
        print("  SUCCESS: model loaded", flush=True)
        lib.audiocpp_free_model(model)
        lib.audiocpp_clear_error(ctypes.byref(err))
        return 0
    else:
        msg = err.message.decode("utf-8", "replace") if err.message else "(null)"
        print(f"  FAILED: code={err.code} msg={msg}", flush=True)
        lib.audiocpp_clear_error(ctypes.byref(err))
        return 1

print(f"audiocpp version: {lib.audiocpp_version().decode()}")
print(f"dll: {DLL}")
print(f"model: {MODEL}")

# 测试1: 带 family_hint
r1 = try_load("TEST 1: with family_hint", "qwen3_asr")
# 测试2: 不带 family_hint (触发 can_load 遍历)
r2 = try_load("TEST 2: family_hint=NULL (auto-detect)", None)

print("\n=== SUMMARY ===")
print(f"  with hint:    {'OK' if r1==0 else 'FAILED/CRASH'}")
print(f"  without hint: {'OK' if r2==0 else 'FAILED/CRASH'}")
sys.exit(1 if (r1 or r2) else 0)
