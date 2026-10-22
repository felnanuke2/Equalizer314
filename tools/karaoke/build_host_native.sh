#!/bin/sh
# Builds the karaoke DSP engine as a host shared library for the desktop real-model tests
# (SpleeterStream runs the same C on Android via the APK's libkaraoke_jni.so; the JVM tests
# load this instead — set KARAOKE_NATIVE_LIB to the result when running gradle tests).
#
# Usage: tools/karaoke/build_host_native.sh [output-dir]   (default: <repo>/build/host-native)
set -e
cd "$(dirname "$0")/../.."
CPP=app/src/main/cpp
OUT="${1:-build/host-native}"
mkdir -p "$OUT"

UNAME="$(uname -s)"
case "$UNAME" in
    Darwin) SYS=linux; INC_DARWIN="-I$JAVA_HOME/include/darwin";;
    *) echo "unsupported host OS: $UNAME" >&2; exit 1;;
esac

if [ -z "$JAVA_HOME" ]; then
    JAVA_HOME="$(dirname "$(dirname "$(javac -which 2>/dev/null || command -v javac)")")"
    export JAVA_HOME
fi
if [ ! -d "$JAVA_HOME/include" ]; then
    echo "JAVA_HOME not set or has no include/ dir (need JDK headers for jni.h)" >&2
    exit 1
fi

ARCH="$(uname -m)"
case "$ARCH" in
    arm64|aarch64) SIMD="-DHAVE_NEON";;
    *) SIMD="-DHAVE_SSE";;
esac

set -x
CC="${CC:-cc}"
$CC -c -O3 -ffp-contract=fast $SIMD -I"$CPP/shim" -I"$CPP/pipewire" \
    "$CPP/pipewire/pffft.c" -o "$OUT/pffft_simd.o" -Wno-unused-function
$CC -c -O3 -ffp-contract=fast $SIMD -DPFFFT_SIMD_DISABLE -I"$CPP/shim" -I"$CPP/pipewire" \
    "$CPP/pipewire/pffft.c" -o "$OUT/pffft_c.o" -Wno-unused-function
$CC -c -O3 -ffp-contract=fast -I"$JAVA_HOME/include" $INC_DARWIN -I"$CPP/shim" -I"$CPP/pipewire" \
    "$CPP/karaoke_jni.c" -o "$OUT/karaoke_jni.o"
$CC -shared -o "$OUT/libkaraoke_jni.dylib" \
    "$OUT/karaoke_jni.o" "$OUT/pffft_simd.o" "$OUT/pffft_c.o" -lm -lpthread
echo "built $OUT/libkaraoke_jni.dylib — run tests with KARAOKE_NATIVE_LIB=$OUT/libkaraoke_jni.dylib"
