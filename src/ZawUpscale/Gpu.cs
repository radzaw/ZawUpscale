using System.Runtime.InteropServices;

namespace ZawUpscale;

public sealed record GpuAdapter(int Index, string Name, long MemoryBytes);

/// <summary>
/// Lists the GPUs of this build's backend, numbered as it numbers them, so a choice in the UI maps to the right
/// device id: DirectX adapters for DirectML (DXGI EnumAdapters1), NVIDIA GPUs for CUDA (NVML, in PCI bus order,
/// which CUDA is told to use too), and the Apple chip for Core ML. Without a GPU backend the list is empty.
/// </summary>
public static unsafe class Gpu
{
    [DllImport("dxgi.dll")]
    static extern int CreateDXGIFactory1(in Guid riid, out IntPtr factory);

    static readonly Guid IID_IDXGIFactory1 = new("770aae78-f26f-4dba-a829-253c83d1b387");
    const int DXGI_ERROR_NOT_FOUND = unchecked((int)0x887A0002);
    const uint DXGI_ADAPTER_FLAG_SOFTWARE = 2;

    [StructLayout(LayoutKind.Sequential)]
    struct AdapterDesc1
    {
        public fixed char Description[128];
        public uint VendorId, DeviceId, SubSysId, Revision;
        public nuint DedicatedVideoMemory, DedicatedSystemMemory, SharedSystemMemory;
        public uint LuidLow;
        public int LuidHigh;
        public uint Flags;
    }

    static List<GpuAdapter>? cache;

    public static List<GpuAdapter> Adapters()
    {
        if (cache != null) return cache;
        var list = new List<GpuAdapter>();
        try
        {
#if DIRECTML
            if (OperatingSystem.IsWindows()) Enumerate(list);
#elif CUDA
            EnumerateNvidia(list);
#elif COREML
            if (OperatingSystem.IsMacOS()) list.Add(AppleChip());
#endif
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or SEHException) { /* no driver */ }
        return cache = list;
    }

    // ---- CUDA: NVML, which comes with the NVIDIA driver.

    const string Nvml = "libnvidia-ml.so.1";
    [DllImport(Nvml)] static extern int nvmlInit_v2();
    [DllImport(Nvml)] static extern int nvmlShutdown();
    [DllImport(Nvml)] static extern int nvmlDeviceGetCount_v2(out uint count);
    [DllImport(Nvml)] static extern int nvmlDeviceGetHandleByIndex_v2(uint index, out IntPtr device);
    [DllImport(Nvml)] static extern int nvmlDeviceGetName(IntPtr device, byte* name, uint length);
    [DllImport(Nvml)] static extern int nvmlDeviceGetMemoryInfo(IntPtr device, out NvmlMemory memory);

    [StructLayout(LayoutKind.Sequential)]
    struct NvmlMemory { public ulong Total, Free, Used; }

    [DllImport("libc", EntryPoint = "setenv")]
    static extern int SetEnv(string name, string value, int overwrite);

    static void EnumerateNvidia(List<GpuAdapter> list)
    {
        // CUDA numbers GPUs fastest first unless told otherwise; NVML by PCI bus. Make CUDA match, before it starts.
        // (Environment.SetEnvironmentVariable doesn't reach native code on Unix, hence libc.)
        if (Environment.GetEnvironmentVariable("CUDA_DEVICE_ORDER") == null) SetEnv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0);
        if (nvmlInit_v2() != 0) return;
        try
        {
            if (nvmlDeviceGetCount_v2(out var count) != 0) return;
            var name = stackalloc byte[96];
            for (uint i = 0; i < count; i++)
            {
                if (nvmlDeviceGetHandleByIndex_v2(i, out var device) != 0) continue;
                var label = nvmlDeviceGetName(device, name, 96) == 0 ? Marshal.PtrToStringUTF8((IntPtr)name) ?? $"GPU {i}" : $"GPU {i}";
                var memory = nvmlDeviceGetMemoryInfo(device, out var m) == 0 ? (long)m.Total : 0;
                list.Add(new GpuAdapter((int)i, label, memory));
            }
        }
        finally
        {
            nvmlShutdown();
        }
    }

    // ---- Core ML: one device, the Apple chip's GPU and Neural Engine (memory is shared with the system, so none is shown).

    [DllImport("libc", EntryPoint = "sysctlbyname")]
    static extern int SysctlByName(string name, byte* value, ref nuint length, IntPtr newValue, nuint newLength);

    static GpuAdapter AppleChip()
    {
        nuint length = 0;
        var chip = "Apple silicon";
        if (SysctlByName("machdep.cpu.brand_string", null, ref length, IntPtr.Zero, 0) == 0 && length > 0)
        {
            var buf = stackalloc byte[(int)length];
            if (SysctlByName("machdep.cpu.brand_string", buf, ref length, IntPtr.Zero, 0) == 0)
                chip = Marshal.PtrToStringUTF8((IntPtr)buf)?.Trim() is { Length: > 0 } s ? s : chip;
        }
        return new GpuAdapter(0, $"{chip} GPU & Neural Engine", 0);
    }

    static void Enumerate(List<GpuAdapter> list)
    {
        if (CreateDXGIFactory1(IID_IDXGIFactory1, out var factory) < 0 || factory == IntPtr.Zero) return;
        try
        {
            // IDXGIFactory1 vtable: IUnknown (3), IDXGIObject (4), IDXGIFactory (5), then EnumAdapters1.
            var fvt = *(IntPtr**)factory;
            var enumAdapters1 = (delegate* unmanaged[Stdcall]<IntPtr, uint, IntPtr*, int>)fvt[12];
            for (uint i = 0; i < 16; i++)
            {
                IntPtr adapter;
                var hr = enumAdapters1(factory, i, &adapter);
                if (hr == DXGI_ERROR_NOT_FOUND || hr < 0) break;
                // IDXGIAdapter1 vtable: IUnknown (3), IDXGIObject (4), IDXGIAdapter (3), then GetDesc1.
                var avt = *(IntPtr**)adapter;
                var getDesc1 = (delegate* unmanaged[Stdcall]<IntPtr, AdapterDesc1*, int>)avt[10];
                var release = (delegate* unmanaged[Stdcall]<IntPtr, uint>)avt[2];
                AdapterDesc1 d;
                if (getDesc1(adapter, &d) >= 0 && (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0)
                    list.Add(new GpuAdapter((int)i, new string(d.Description).TrimEnd('\0').Trim(), (long)d.DedicatedVideoMemory));
                release(adapter);
            }
        }
        finally
        {
            var release = (delegate* unmanaged[Stdcall]<IntPtr, uint>)(*(IntPtr**)factory)[2];
            release(factory);
        }
    }
}
