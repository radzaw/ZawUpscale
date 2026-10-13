# Renders the ZawUpscale icon (the same drawing as zawupscale.svg) into PNGs, the Windows .ico and the web app icon.
# Uses GDI+ (System.Drawing), so it runs on Windows PowerShell without other tools:
#   powershell -ExecutionPolicy Bypass -File assets/icon/build-icons.ps1
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$root = Resolve-Path (Join-Path $PSScriptRoot '../..')
$pngDir = Join-Path $PSScriptRoot 'png'
New-Item -ItemType Directory -Force $pngDir | Out-Null

function Rounded([float]$x, [float]$y, [float]$w, [float]$h, [float]$r) {
    $d = 2 * $r
    $p = New-Object System.Drawing.Drawing2D.GraphicsPath
    $p.AddArc($x, $y, $d, $d, 180, 90); $p.AddArc($x + $w - $d, $y, $d, $d, 270, 90)
    $p.AddArc($x + $w - $d, $y + $h - $d, $d, $d, 0, 90); $p.AddArc($x, $y + $h - $d, $d, $d, 90, 90)
    $p.CloseFigure()
    return , $p
}

# A quadratic Bezier (as in the SVG) added to a path as the equivalent cubic.
function Quad($path, [float]$x0, [float]$y0, [float]$cx, [float]$cy, [float]$x1, [float]$y1) {
    $path.AddBezier($x0, $y0, ($x0 + 2 * ($cx - $x0) / 3), ($y0 + 2 * ($cy - $y0) / 3), ($x1 + 2 * ($cx - $x1) / 3), ($y1 + 2 * ($cy - $y1) / 3), $x1, $y1)
}

function Render([int]$size) {
    $bmp = New-Object System.Drawing.Bitmap $size, $size, ([System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = 'AntiAlias'
    $g.PixelOffsetMode = 'HighQuality'
    $g.Clear([System.Drawing.Color]::Transparent)
    $k = $size / 512.0
    $g.ScaleTransform($k, $k)

    # Rounded square with the diagonal gradient.
    $brush = New-Object System.Drawing.Drawing2D.LinearGradientBrush (New-Object System.Drawing.Point 0, 0), (New-Object System.Drawing.Point 512, 512), ([System.Drawing.Color]::Black), ([System.Drawing.Color]::Black)
    $blend = New-Object System.Drawing.Drawing2D.ColorBlend 3
    $blend.Colors = @([System.Drawing.ColorTranslator]::FromHtml('#a3e635'), [System.Drawing.ColorTranslator]::FromHtml('#10b981'), [System.Drawing.ColorTranslator]::FromHtml('#0e7490'))
    $blend.Positions = @(0.0, 0.5, 1.0)
    $brush.InterpolationColors = $blend
    $g.FillPath($brush, (Rounded 0 0 512 512 112))

    # Coarse pixels.
    foreach ($px in @(@(104, 280, 255), @(172, 280, 184), @(104, 348, 184), @(172, 348, 122))) {
        $c = [System.Drawing.Color]::FromArgb($px[2], 255, 255, 255)
        $g.FillPath((New-Object System.Drawing.SolidBrush $c), (Rounded $px[0] $px[1] 60 60 10))
    }

    # Arrow.
    $pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::White), 40
    $pen.StartCap = 'Round'; $pen.EndCap = 'Round'; $pen.LineJoin = 'Round'
    $g.DrawLine($pen, 262, 250, 392, 120)
    $g.DrawLines($pen, [System.Drawing.PointF[]]@((New-Object System.Drawing.PointF 300, 120), (New-Object System.Drawing.PointF 392, 120), (New-Object System.Drawing.PointF 392, 212)))

    # Sparkle.
    $s = New-Object System.Drawing.Drawing2D.GraphicsPath
    Quad $s 362 282 368 350 436 356
    Quad $s 436 356 368 362 362 430
    Quad $s 362 430 356 362 288 356
    Quad $s 288 356 356 350 362 282
    $s.CloseFigure()
    $g.FillPath((New-Object System.Drawing.SolidBrush ([System.Drawing.ColorTranslator]::FromHtml('#ecfccb'))), $s)

    $g.Dispose()
    $ms = New-Object System.IO.MemoryStream
    $bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    return , $ms.ToArray()
}

$pngs = @{}
foreach ($s in 16, 20, 24, 32, 40, 48, 64, 128, 256, 512) {
    $pngs[$s] = Render $s
    if ($s -notin 20, 40) { [System.IO.File]::WriteAllBytes((Join-Path $pngDir "zawupscale-$s.png"), $pngs[$s]) }
}
[System.IO.File]::WriteAllBytes((Join-Path $root 'src/ZawUpscale/wwwroot/icon-256.png'), $pngs[256])

# Windows icon: PNG-compressed entries (supported since Windows Vista) at the sizes Explorer and the taskbar use.
$icoSizes = 16, 20, 24, 32, 40, 48, 64, 256
$out = New-Object System.IO.MemoryStream
$w = New-Object System.IO.BinaryWriter $out
$w.Write([uint16]0); $w.Write([uint16]1); $w.Write([uint16]$icoSizes.Count)
$offset = 6 + 16 * $icoSizes.Count
foreach ($s in $icoSizes) {
    $dim = if ($s -ge 256) { 0 } else { $s }
    $w.Write([byte]$dim); $w.Write([byte]$dim); $w.Write([byte]0); $w.Write([byte]0)
    $w.Write([uint16]1); $w.Write([uint16]32)
    $w.Write([uint32]$pngs[$s].Length); $w.Write([uint32]$offset)
    $offset += $pngs[$s].Length
}
foreach ($s in $icoSizes) { $w.Write($pngs[$s]) }
$w.Flush()
[System.IO.File]::WriteAllBytes((Join-Path $root 'src/ZawUpscale/zawupscale.ico'), $out.ToArray())
Write-Host "Icons written to $pngDir, src/ZawUpscale/zawupscale.ico and src/ZawUpscale/wwwroot/icon-256.png"
