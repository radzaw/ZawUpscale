using System.Globalization;
using System.Runtime.InteropServices;

namespace ZawUpscale;

/// <summary>
/// The Windows taskbar button: progress (green, yellow when paused, red after an error) through ITaskbarList3, and
/// flashing to ask for attention. Called on the window's (STA) thread.
/// </summary>
static class Taskbar
{
    [ComImport, Guid("ea1afb91-9e28-4b86-90e9-9e9f8a5eefaf"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface ITaskbarList3
    {
        // ITaskbarList
        void HrInit();
        void AddTab(IntPtr hwnd);
        void DeleteTab(IntPtr hwnd);
        void ActivateTab(IntPtr hwnd);
        void SetActiveAlt(IntPtr hwnd);
        // ITaskbarList2
        void MarkFullscreenWindow(IntPtr hwnd, [MarshalAs(UnmanagedType.Bool)] bool fullscreen);
        // ITaskbarList3 (only the two used here; the rest of the vtable follows them)
        void SetProgressValue(IntPtr hwnd, ulong completed, ulong total);
        void SetProgressState(IntPtr hwnd, int state);
    }

    [ComImport, Guid("56fdf344-fd6d-11d0-958a-006097c9a090"), ClassInterface(ClassInterfaceType.None)]
    class TaskbarInstance;

    const int NoProgress = 0, Normal = 2, Error = 4, Paused = 8;
    static ITaskbarList3? list;
    static bool failed;

    /// <summary>"0.42" (a fraction done), "paused", "error" or "none".</summary>
    public static void SetProgress(IntPtr hwnd, string value)
    {
        if (hwnd == IntPtr.Zero || Instance() is not { } t) return;
        try
        {
            switch (value)
            {
                case "none": t.SetProgressState(hwnd, NoProgress); break;
                case "paused": t.SetProgressState(hwnd, Paused); break;
                case "error": t.SetProgressState(hwnd, Error); break;
                default:
                    if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var f)) return;
                    t.SetProgressState(hwnd, Normal);
                    t.SetProgressValue(hwnd, (ulong)Math.Round(Math.Clamp(f, 0, 1) * 1000), 1000);
                    break;
            }
        }
        catch (COMException) { /* the taskbar (Explorer) restarted or isn't there */ }
    }

    static ITaskbarList3? Instance()
    {
        if (list != null || failed) return list;
        try
        {
            var t = (ITaskbarList3)new TaskbarInstance();
            t.HrInit();
            return list = t;
        }
        catch (Exception ex) when (ex is COMException or InvalidCastException)
        {
            failed = true;
            return null;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    struct FlashInfo
    {
        public uint Size;
        public IntPtr Hwnd;
        public uint Flags;
        public uint Count;
        public uint Timeout;
    }

    [DllImport("user32.dll")]
    static extern bool FlashWindowEx(ref FlashInfo info);

    /// <summary>Flashes the taskbar button until the window comes to the front.</summary>
    public static void Flash(IntPtr hwnd)
    {
        if (hwnd == IntPtr.Zero) return;
        var info = new FlashInfo { Size = (uint)Marshal.SizeOf<FlashInfo>(), Hwnd = hwnd, Flags = 2 /* FLASHW_TRAY */ | 12 /* FLASHW_TIMERNOFG */, Count = 0, Timeout = 0 };
        FlashWindowEx(ref info);
    }
}
