"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput.  With the AMD backend (#301): the
  amdgpu driver's Linux sysfs files - load, VRAM, temperature and power.
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  the OS (Windows GlobalMemoryStatusEx / GetSystemTimes, Linux /proc) and the disk rate is absent.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import os
import platform
import sys
import threading
import time

HISTORY = 60


# ------------------------------------------------------------------------------------------------ NVML
class _Nvml:
    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, index=0):
        self.lib = self.dev = None
        names = ["nvml.dll", os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                          "NVIDIA Corporation", "NVSMI", "nvml.dll")] if os.name == "nt" \
            else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        for n in names:
            try:
                self.lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if self.lib is None:
            return
        try:
            init = getattr(self.lib, "nvmlInit_v2", None) or self.lib.nvmlInit
            if init() != 0:
                self.lib = None
                return
            h = ctypes.c_void_p()
            get = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", None) or self.lib.nvmlDeviceGetHandleByIndex
            if get(ctypes.c_uint(index), ctypes.byref(h)) != 0:
                self.lib = None
                return
            self.dev = h
        except (AttributeError, OSError):
            self.lib = None

    def ok(self):
        return self.lib is not None and self.dev is not None

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, *args, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def name(self):
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace")
        except (AttributeError, OSError):
            pass
        return None

    def read(self):
        out = {}
        u = self.Util()
        try:
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, ctypes.byref(u)) == 0:
                out["util"] = u.gpu
        except (AttributeError, OSError):
            pass
        m = self.Mem()
        try:
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0:
                out["mem_used"], out["mem_total"] = m.used, m.total
        except (AttributeError, OSError):
            pass
        out["temp"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))          # NVML_TEMPERATURE_GPU
        mw = self._uint("nvmlDeviceGetPowerUsage")
        out["power"] = mw / 1000.0 if mw is not None else None
        lim = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        out["power_limit"] = lim / 1000.0 if lim is not None else None
        out["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")        # drops at idle (power saving)
        out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        rx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(1))                 # NVML_PCIE_UTIL_RX_BYTES, KB/s
        tx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(0))
        out["pcie_rx_mb"] = rx / 1024.0 if rx is not None else None
        out["pcie_tx_mb"] = tx / 1024.0 if tx is not None else None
        return out


# ------------------------------------------------------------------------------------------------ AMD (Linux sysfs)
SYSFS = "/sys"
WINDOWS = os.name == "nt"


def amd_device_dir(index, sysfs=None):
    """The amdgpu sysfs folder (/sys/class/drm/renderD<N>/device) of the AMD GPU that HIP numbers `index`: the KFD
    topology's GPU nodes in order, the CPU nodes skipped, linked to their render node by drm_render_minor - the
    numbering setup's amd_gpus() and HIP_VISIBLE_DEVICES use.  None when there is no such card (or no amdgpu)."""
    base = os.path.join(sysfs or SYSFS, "class", "kfd", "kfd", "topology", "nodes")
    try:
        nodes = sorted((n for n in os.listdir(base) if n.isdigit()), key=int)
    except OSError:
        return None
    gpus = []
    for n in nodes:
        try:
            with open(os.path.join(base, n, "properties"), encoding="utf-8") as f:
                props = dict(line.strip().partition(" ")[::2] for line in f if line.strip())
            if int(props.get("gfx_target_version") or 0) == 0 or int(props.get("simd_count") or 0) == 0:
                continue
            gpus.append(props)
        except (OSError, ValueError):
            continue
    if not 0 <= index < len(gpus) or not gpus[index].get("drm_render_minor"):
        return None
    dev = os.path.join(sysfs or SYSFS, "class", "drm", "renderD" + gpus[index]["drm_render_minor"].strip(), "device")
    return dev if os.path.isdir(dev) else None


class _Amd:
    """#301: an AMD card's readings from the amdgpu driver's sysfs files (Linux; no ROCm library needed), with _Nvml's
    interface: load (gpu_busy_percent), VRAM (mem_info_vram_used / _total), and from its hwmon folder the temperature
    (temp1_input, the edge sensor, m°C), power (power1_average or power1_input, µW) and its cap (power1_cap)."""

    def __init__(self, index=0, sysfs=None):
        self.dev = amd_device_dir(index, sysfs)
        self.hwmon = None
        if self.dev:
            try:
                hw = sorted(os.listdir(os.path.join(self.dev, "hwmon")))
                self.hwmon = os.path.join(self.dev, "hwmon", hw[0]) if hw else None
            except OSError:
                pass

    def ok(self):
        return self.dev is not None

    @staticmethod
    def _int(path):
        try:
            with open(path, encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError, TypeError):
            return None

    def name(self):
        try:
            with open(os.path.join(self.dev, "product_name"), encoding="utf-8") as f:
                return f.read().strip() or "AMD Radeon"
        except (OSError, TypeError):
            return "AMD Radeon"

    def read(self):
        out = {"util": self._int(os.path.join(self.dev, "gpu_busy_percent")),
               "mem_used": self._int(os.path.join(self.dev, "mem_info_vram_used")),
               "mem_total": self._int(os.path.join(self.dev, "mem_info_vram_total"))}
        if self.hwmon:
            t = self._int(os.path.join(self.hwmon, "temp1_input"))
            out["temp"] = t / 1000.0 if t is not None else None
            p = self._int(os.path.join(self.hwmon, "power1_average"))
            if p is None:
                p = self._int(os.path.join(self.hwmon, "power1_input"))
            out["power"] = p / 1e6 if p is not None else None
            cap = self._int(os.path.join(self.hwmon, "power1_cap"))
            out["power_limit"] = cap / 1e6 if cap is not None else None
        return out


# ------------------------------------------------------------------------------------------------ AMD (Windows)
def _dxgi_adapters(vendor=0x1002):
    """[(name, dedicated VRAM bytes, "luid_0x<High>_0x<Low>", as the counters name it)] of the DXGI adapters of one vendor (AMD by default), most
    VRAM first (as HIP numbers them), software adapters skipped.  Empty off Windows or when DXGI fails."""
    import ctypes.wintypes as wt

    class Luid(ctypes.Structure):
        _fields_ = [("Low", wt.DWORD), ("High", wt.LONG)]

    class Desc1(ctypes.Structure):
        _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", wt.UINT), ("DeviceId", wt.UINT),
                    ("SubSysId", wt.UINT), ("Revision", wt.UINT), ("Dedicated", ctypes.c_size_t),
                    ("DedicatedSys", ctypes.c_size_t), ("Shared", ctypes.c_size_t), ("Luid", Luid),
                    ("Flags", wt.UINT)]

    class Guid(ctypes.Structure):
        _fields_ = [("a", wt.DWORD), ("b", wt.WORD), ("c", wt.WORD), ("d", ctypes.c_ubyte * 8)]

    def method(obj, index, *argtypes):
        vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtbl[index])

    out = []
    try:
        iid = Guid(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
        fac = ctypes.c_void_p()
        if ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(fac)) != 0 or not fac.value:
            return out
        i = 0
        while True:                                                       # IDXGIFactory1::EnumAdapters1 (slot 12)
            ad = ctypes.c_void_p()
            if method(fac, 12, wt.UINT, ctypes.POINTER(ctypes.c_void_p))(fac, i, ctypes.byref(ad)) != 0:
                break
            i += 1
            d = Desc1()
            if method(ad, 10, ctypes.POINTER(Desc1))(ad, ctypes.byref(d)) == 0 \
                    and d.VendorId == vendor and not d.Flags & 2:       # IDXGIAdapter1::GetDesc1; 2 = SOFTWARE
                out.append((d.Description, int(d.Dedicated), f"luid_0x{d.Luid.High & 0xFFFFFFFF:08X}_0x{d.Luid.Low:08X}"))
            method(ad, 2)(ad)                                             # Release
        method(fac, 2)(fac)
    except (OSError, AttributeError, ValueError):
        pass
    return sorted(out, key=lambda a: -a[1])          # HIP lists the big cards first and the integrated one last


class _Pdh:
    """Windows performance counters through pdh.dll: wildcard counter -> {instance name: value}."""

    def __init__(self):
        import ctypes.wintypes as wt
        self.wt = wt
        p = ctypes.windll.pdh
        self.p = p
        p.PdhOpenQueryW.argtypes = [wt.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wt.HANDLE)]
        p.PdhAddEnglishCounterW.argtypes = [wt.HANDLE, wt.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wt.HANDLE)]
        p.PdhCollectQueryData.argtypes = [wt.HANDLE]
        p.PdhGetFormattedCounterArrayW.argtypes = [wt.HANDLE, wt.DWORD, ctypes.POINTER(wt.DWORD),
                                                   ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
        self.q = wt.HANDLE()
        if p.PdhOpenQueryW(None, 0, ctypes.byref(self.q)):
            raise OSError("PdhOpenQuery")
        self.counters = {}

    def add(self, name, path):
        c = self.wt.HANDLE()
        if self.p.PdhAddEnglishCounterW(self.q, path, 0, ctypes.byref(c)):
            raise OSError("PdhAddCounter " + path)
        self.counters[name] = c

    def collect(self):
        self.p.PdhCollectQueryData(self.q)

    def values(self, name):
        wt = self.wt

        class Item(ctypes.Structure):
            _fields_ = [("name", wt.LPWSTR), ("status", wt.DWORD), ("value", ctypes.c_double)]

        c, size, n = self.counters[name], wt.DWORD(0), wt.DWORD(0)
        self.p.PdhGetFormattedCounterArrayW(c, 0x200, ctypes.byref(size), ctypes.byref(n), None)   # PDH_FMT_DOUBLE
        if not size.value:
            return {}
        buf = ctypes.create_string_buffer(size.value)
        if self.p.PdhGetFormattedCounterArrayW(c, 0x200, ctypes.byref(size), ctypes.byref(n), buf):
            return {}
        # PDH_FMT_COUNTERVALUE_ITEM_W: LPWSTR name, then a PDH_FMT_COUNTERVALUE {DWORD status; double value}
        out = {}
        for j in range(n.value):
            it = Item.from_buffer_copy(buf, j * ctypes.sizeof(Item))
            if it.status in (0, 1):
                out[it.name] = it.value
        return out


class _Adl:
    """AMD's display library (atiadlxx.dll, installed with the Radeon driver): the cards in PCI-bus order with their
    adapter index, and one ADL2_New_QueryPMLogData_Get call per card for all its sensors at once.  Shared by every
    reader; `lib` is None when the library is missing."""

    SENSORS = 256
    TEMP_EDGE, ACTIVITY_GFX, BOARD_POWER = 8, 19, 73                     # ADL_PMLOG_SENSORS indices

    class Info(ctypes.Structure):
        _fields_ = [("Size", ctypes.c_int), ("AdapterIndex", ctypes.c_int), ("UDID", ctypes.c_char * 256),
                    ("BusNumber", ctypes.c_int), ("DeviceNumber", ctypes.c_int), ("FunctionNumber", ctypes.c_int),
                    ("VendorID", ctypes.c_int), ("AdapterName", ctypes.c_char * 256),
                    ("DisplayName", ctypes.c_char * 256), ("Present", ctypes.c_int), ("Exist", ctypes.c_int),
                    ("DriverPath", ctypes.c_char * 256), ("DriverPathExt", ctypes.c_char * 256),
                    ("PNPString", ctypes.c_char * 256), ("OSDisplayIndex", ctypes.c_int)]

    class Sensor(ctypes.Structure):
        _fields_ = [("supported", ctypes.c_int), ("value", ctypes.c_int)]

    class PmLog(ctypes.Structure):
        pass

    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        self.lib = self.ctx = None
        self.cards = []                                                  # [(adapter index, bus, device)]
        self.PmLog._fields_ = [("size", ctypes.c_int), ("sensors", self.Sensor * self.SENSORS)]
        try:
            lib = ctypes.CDLL("atiadlxx.dll")
            self._alloc = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)(ctypes.cdll.msvcrt.malloc)
            ctx = ctypes.c_void_p()
            if lib.ADL2_Main_Control_Create(self._alloc, 1, ctypes.byref(ctx)) != 0:
                return
            n = ctypes.c_int()
            if lib.ADL2_Adapter_NumberOfAdapters_Get(ctx, ctypes.byref(n)) != 0 or n.value <= 0:
                return
            arr = (self.Info * n.value)()
            if lib.ADL2_Adapter_AdapterInfo_Get(ctx, arr, ctypes.c_int(ctypes.sizeof(arr))) != 0:
                return
            seen, cards = set(), []
            for a in arr:                                                # one entry per adapter, not per display
                if a.VendorID == 1002 and a.Present and (a.BusNumber, a.DeviceNumber) not in seen:
                    seen.add((a.BusNumber, a.DeviceNumber))
                    cards.append((a.AdapterIndex, a.BusNumber, a.DeviceNumber))
            self.lib, self.ctx, self.cards = lib, ctx, sorted(cards, key=lambda c: c[1:])
        except (OSError, AttributeError, ValueError):
            self.lib = None

    def pmlog(self, adapter):
        """{sensor index: value} of the supported sensors, or None."""
        if self.lib is None:
            return None
        try:
            o = self.PmLog()
            o.size = ctypes.sizeof(o)
            if self.lib.ADL2_New_QueryPMLogData_Get(self.ctx, adapter, ctypes.byref(o)) != 0:
                return None
            return {i: o.sensors[i].value for i in range(self.SENSORS) if o.sensors[i].supported}
        except (OSError, AttributeError, ValueError):
            return None


def _pcie_link(bus, device):
    """(generation, width, highest generation) of the PCIe link of the display adapter at this bus/device, or Nones: the
    CurrentLinkSpeed / CurrentLinkWidth / MaxLinkSpeed device properties Windows keeps for every PCIe device (SetupAPI)."""
    import ctypes.wintypes as wt

    class Guid(ctypes.Structure):
        _fields_ = [("a", wt.DWORD), ("b", wt.WORD), ("c", wt.WORD), ("d", ctypes.c_ubyte * 8)]

    class DevInfo(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("cls", Guid), ("inst", wt.DWORD), ("res", ctypes.c_void_p)]

    class PropKey(ctypes.Structure):
        _fields_ = [("fmtid", Guid), ("pid", wt.DWORD)]

    try:
        sa = ctypes.WinDLL("setupapi", use_last_error=True)
        sa.SetupDiGetClassDevsW.restype = ctypes.c_void_p
        sa.SetupDiGetClassDevsW.argtypes = [ctypes.POINTER(Guid), wt.LPCWSTR, wt.HWND, wt.DWORD]
        sa.SetupDiEnumDeviceInfo.argtypes = [ctypes.c_void_p, wt.DWORD, ctypes.POINTER(DevInfo)]
        sa.SetupDiGetDeviceRegistryPropertyW.argtypes = [ctypes.c_void_p, ctypes.POINTER(DevInfo), wt.DWORD,
                                                         ctypes.POINTER(wt.DWORD), ctypes.c_void_p, wt.DWORD,
                                                         ctypes.POINTER(wt.DWORD)]
        sa.SetupDiGetDevicePropertyW.argtypes = [ctypes.c_void_p, ctypes.POINTER(DevInfo), ctypes.POINTER(PropKey),
                                                 ctypes.POINTER(wt.DWORD), ctypes.c_void_p, wt.DWORD,
                                                 ctypes.POINTER(wt.DWORD), wt.DWORD]
        sa.SetupDiDestroyDeviceInfoList.argtypes = [ctypes.c_void_p]
        display = Guid(0x4D36E968, 0xE325, 0x11CE, (ctypes.c_ubyte * 8)(0xBF, 0xC1, 0x08, 0x00, 0x2B, 0xE1, 0x03, 0x18))
        h = sa.SetupDiGetClassDevsW(ctypes.byref(display), None, None, 2)          # DIGCF_PRESENT
        if h in (None, ctypes.c_void_p(-1).value):
            return None, None, None
        pci = Guid(0x3AB22E31, 0x8264, 0x4B4E, (ctypes.c_ubyte * 8)(0x9A, 0xF5, 0xA8, 0xD2, 0xD8, 0xE3, 0x3E, 0x62))

        def prop(info, pid):                                                       # DEVPKEY_PciDevice_* (uint32)
            key, v, t = PropKey(pci, pid), wt.DWORD(), wt.DWORD()
            ok = sa.SetupDiGetDevicePropertyW(h, ctypes.byref(info), ctypes.byref(key), ctypes.byref(t),
                                              ctypes.byref(v), 4, None, 0)
            return v.value if ok else None

        def reg(info, code):                                                       # SPDRP_BUSNUMBER 0x15, _ADDRESS 0x1C
            v = wt.DWORD()
            ok = sa.SetupDiGetDeviceRegistryPropertyW(h, ctypes.byref(info), code, None, ctypes.byref(v), 4, None)
            return v.value if ok else None
        try:
            i = 0
            while True:
                info = DevInfo()
                info.cb = ctypes.sizeof(info)
                if not sa.SetupDiEnumDeviceInfo(h, i, ctypes.byref(info)):
                    return None, None, None
                i += 1
                addr = reg(info, 0x1C)                                             # (device << 16) | function
                if reg(info, 0x15) == bus and addr is not None and addr >> 16 == device:
                    speed, width, top = prop(info, 9), prop(info, 10), prop(info, 11)
                    return (speed if speed and speed < 16 else None), (width or None), (top if top and top < 16 else None)
        finally:
            sa.SetupDiDestroyDeviceInfoList(h)
    except (OSError, AttributeError, ValueError):
        return None, None, None


class _AmdWin:
    """AMD card on Windows, with _Nvml's interface.  Name and VRAM size from DXGI; from AMD's ADL library (one
    PMLog call per reading): load, temperature (edge) and board power; VRAM in use from the "GPU Adapter Memory"
    counter; the PCIe link generation and width from SetupAPI.  Without ADL the load comes from the (much costlier)
    "GPU Engine" counters, as Task Manager does it.  PCIe traffic and a power limit are not readable on Windows.
    The cards are numbered as HIP numbers them: the big ones first, in PCI-bus order, the integrated one last."""

    def __init__(self, index=0):
        self.info = self.pdh = self.adl = self.slot = None
        try:
            cards = _dxgi_adapters()
            if not 0 <= index < len(cards):
                return
            pdh = _Pdh()
            pdh.add("mem", r"\GPU Adapter Memory(*)\Dedicated Usage")
            adl = _Adl.get()
            if len(adl.cards) == len(cards):                 # both see the same cards: pair them by position
                self.slot = adl.cards[index]
                if adl.pmlog(self.slot[0]) is not None:
                    self.adl = adl
            if self.adl is None:
                pdh.add("eng", r"\GPU Engine(*)\Utilization Percentage")
            pdh.collect()
            self.pdh, self.info = pdh, cards[index]
        except (OSError, AttributeError, ValueError):
            self.info = self.pdh = None

    def ok(self):
        return self.info is not None and self.pdh is not None

    def name(self):
        return self.info[0]

    def read(self):
        out = {"util": None, "mem_used": None, "mem_total": self.info[1]}
        try:
            luid = self.info[2]
            self.pdh.collect()
            used = [v for k, v in self.pdh.values("mem").items() if luid in k]
            out["mem_used"] = int(sum(used)) if used else None
            if self.adl:
                pm = self.adl.pmlog(self.slot[0]) or {}
                for key, sensor in (("util", _Adl.ACTIVITY_GFX), ("temp", _Adl.TEMP_EDGE), ("power", _Adl.BOARD_POWER)):
                    out[key] = float(pm[sensor]) if sensor in pm else None
                out["pcie_gen"], out["pcie_width"], out["pcie_gen_max"] = _pcie_link(self.slot[1], self.slot[2])
            else:
                per_type = {}                              # Task Manager: the busiest engine type, each type's sum
                for k, v in self.pdh.values("eng").items():
                    if luid in k:
                        t = k.rpartition("_engtype_")[2]
                        per_type[t] = per_type.get(t, 0.0) + v
                out["util"] = min(100.0, max(per_type.values())) if per_type else None
        except (OSError, AttributeError, ValueError):
            pass
        return out


def gpu_reader(index=0, amd=False):
    """The card's readings: NVML (NVIDIA), or with the AMD backend (#301) the amdgpu sysfs files (Linux) / DXGI and
    performance counters (Windows)."""
    if amd:
        return _AmdWin(index) if WINDOWS else _Amd(index)
    return _Nvml(index)


def free_vram_mib(index=0, amd=False):
    """Free VRAM of a card in MiB, or None when it cannot be read."""
    g = gpu_reader(index, amd)
    if not g.ok():
        return None
    r = g.read()
    if r.get("mem_total") is None or r.get("mem_used") is None:
        return None
    return int((r["mem_total"] - r["mem_used"]) >> 20)


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_name():
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self):
        self.prev = self._times()

    def _times(self):
        if os.name == "nt":
            idle, kern, user = (ctypes.c_ulonglong() for _ in range(3))
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return idle.value, kern.value + user.value           # kernel time includes idle
            return None
        try:
            f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            return f[3] + f[4], sum(f)
        except (OSError, ValueError):
            return None

    def cpu(self):
        cur = self._times()
        prev, self.prev = self.prev, cur
        if not cur or not prev or cur[1] == prev[1]:
            return None
        return max(0.0, min(100.0, 100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1]))))

    @staticmethod
    def ram():
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullTotalPhys - m.ullAvailPhys, m.ullTotalPhys
            return None, None
        try:
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


# ------------------------------------------------------------------------------------------------ the sampler
class Telemetry:
    def __init__(self, extra=None, gpu_index=0, gpu_indices=None, amd=False):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_index`: the card the
        engine runs on, numbered as nvidia-smi and NVML number them (by PCI bus); `gpu_indices`: all of them when
        the model is split across several (issue #112) - the gpu_* readings are then their total (memory, power,
        PCIe traffic), mean (load) or hottest (temperature), and "gpus" has each card's own.  `amd`: the AMD backend's
        cards, numbered as HIP numbers them, read from sysfs (#301)."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        idx = list(gpu_indices) if gpu_indices and len(gpu_indices) > 1 else [gpu_index]
        self.gpus = [(i, gpu_reader(i, amd)) for i in idx]
        self.gpus = [(i, g) for i, g in self.gpus if g.ok()] or self.gpus[:1]
        self.gpu = self.gpus[0][1]
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        self.static = {
            "gpu_name": " + ".join(g.name() or "?" for _, g in self.gpus) if self.gpu.ok() else None,
            "gpu_count": len(self.gpus),
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "psutil": self.ps is not None,
        }
        self._disk_prev = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _disk(self):
        if not self.ps:
            return None, None
        try:
            c = self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None, None
        t = time.time()
        prev, self._disk_prev = self._disk_prev, (t, c.read_bytes, c.write_bytes)
        if prev is None or t <= prev[0]:
            return None, None
        dt = t - prev[0]
        return (c.read_bytes - prev[1]) / dt / 2**20, (c.write_bytes - prev[2]) / dt / 2**20

    def sample(self):
        s = {}
        if self.gpu.ok():
            reads = [(i, g.read()) for i, g in self.gpus]
            g = dict(reads[0][1])
            if len(reads) > 1:
                def vals(k):
                    return [r[k] for _, r in reads if r.get(k) is not None]
                for k in ("mem_used", "mem_total", "power", "power_limit", "pcie_rx_mb", "pcie_tx_mb"):
                    v = vals(k)
                    g[k] = sum(v) if v else None
                u = vals("util")
                g["util"] = sum(u) / len(u) if u else None
                t = vals("temp")
                g["temp"] = max(t) if t else None
                s["gpus"] = [{"index": i, "util": r.get("util"), "mem_used": r.get("mem_used"),
                              "mem_total": r.get("mem_total"), "temp": r.get("temp"), "power": r.get("power")}
                             for i, r in reads]
            s.update({f"gpu_{k}": v for k, v in g.items()})
        if self.ps:
            try:
                s["cpu"] = self.ps.cpu_percent(interval=None)
                vm = self.ps.virtual_memory()
                s["ram_used"], s["ram_total"] = vm.total - vm.available, vm.total
            except (OSError, RuntimeError):
                pass
        else:
            s["cpu"] = self.fallback.cpu()
            s["ram_used"], s["ram_total"] = self.fallback.ram()
        s["disk_read_mb"], s["disk_write_mb"] = self._disk()
        if self.extra:
            try:
                s.update(self.extra())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        return s

    def _loop(self):
        while True:
            s = self.sample()
            with self.lock:
                self.now = s
                for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used",
                          "disk_read_mb", "tok_s", "prefill_tok_s_mean"):
                    v = s.get(k)
                    self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            time.sleep(1.0)

    def snapshot(self):
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}
