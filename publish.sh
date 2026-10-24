#!/usr/bin/env sh
# Builds self-contained single-file executables of ZawUpscale.
# Usage: ./publish.sh [rid ...]      e.g. ./publish.sh linux-x64 win-x64
# linux-x64-cuda is linux-x64 with ONNX Runtime's CUDA build, for NVIDIA GPUs (much larger; needs CUDA 12 and cuDNN 9).
# Set VERSION (e.g. VERSION=1.2.0) to stamp the build; otherwise the project's version is used.
set -e
cd "$(dirname "$0")"
RIDS=${*:-"win-x64 linux-x64 linux-arm64 osx-arm64"}
for target in $RIDS; do
  rid=${target%-cuda}
  cuda=""
  if [ "$rid" != "$target" ]; then cuda="-p:ZawCuda=true"; fi
  echo "Publishing $target -> dist/$target${VERSION:+ (version $VERSION)}"
  dotnet publish src/ZawUpscale/ZawUpscale.csproj -c Release -r "$rid" -o "dist/$target" $cuda \
    --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true -p:DebugType=none \
    ${VERSION:+-p:Version=$VERSION}
done
