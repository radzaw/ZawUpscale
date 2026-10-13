using System.Runtime.InteropServices;

namespace ZawUpscale;

public sealed record GpuAdapter(int Index, string Name, long MemoryBytes);

/// <summary>
/// Lists the DirectX graphics adapters, in the order DirectML numbers them (DXGI EnumAdapters1),
/// so a choice in the UI maps to the right device id. Windows only; elsewhere the list is empty.
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
        if (OperatingSystem.IsWindows())
        {
            try { Enumerate(list); }
            catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or SEHException) { /* no DXGI */ }
        }
        return cache = list;
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
