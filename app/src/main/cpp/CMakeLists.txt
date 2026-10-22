cmake_minimum_required(VERSION 3.22.1)
project(eq314hrir C)

# pipewire/ holds PipeWire's convolver.c/.h and pffft.c/.h unchanged; shim/ stands in for the PipeWire headers they include.
set(PW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pipewire)
set(SHIM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shim)

if(ANDROID_ABI MATCHES "^(arm64-v8a|armeabi-v7a)$")
    set(SIMD_DEF HAVE_NEON)
else()
    set(SIMD_DEF HAVE_SSE)
endif()

# No compiler banner in .comment: the Windows and Linux NDK clangs word it differently, which breaks F-Droid's byte-for-byte check.
add_compile_options(-fno-ident)

# Real-time DSP: optimise in debug builds too.
set(DSP_FLAGS -O3 -ffp-contract=fast)

# pffft is built twice like PipeWire's meson does: the SIMD kernels, then the scalar build carrying the public API + CPU pick.
add_library(pffft_simd OBJECT ${PW_DIR}/pffft.c)
target_compile_definitions(pffft_simd PRIVATE ${SIMD_DEF})
target_include_directories(pffft_simd PRIVATE ${SHIM_DIR} ${PW_DIR})
target_compile_options(pffft_simd PRIVATE ${DSP_FLAGS} -Wno-unused-function)

add_library(pffft_c OBJECT ${PW_DIR}/pffft.c)
target_compile_definitions(pffft_c PRIVATE PFFFT_SIMD_DISABLE ${SIMD_DEF})
target_include_directories(pffft_c PRIVATE ${SHIM_DIR} ${PW_DIR})
target_compile_options(pffft_c PRIVATE ${DSP_FLAGS} -Wno-unused-function)

add_library(eq314hrir SHARED
    hrir_jni.c
    ${PW_DIR}/convolver.c
    $<TARGET_OBJECTS:pffft_simd>
    $<TARGET_OBJECTS:pffft_c>)
target_include_directories(eq314hrir PRIVATE ${SHIM_DIR} ${PW_DIR})
target_compile_options(eq314hrir PRIVATE ${DSP_FLAGS})
target_link_libraries(eq314hrir m)
# No build id: keeps the .so byte-identical across machines (F-Droid reproducible builds).
target_link_options(eq314hrir PRIVATE -Wl,--build-id=none)
# 16 KB page alignment: Android 15+ phones with 16 KB pages can't load a 4 KB-aligned library (NDK r27 defaults to 4 KB).
target_link_options(eq314hrir PRIVATE -Wl,-z,max-page-size=16384)

# Karaoke separator DSP (STFT/rings/masks/WOLA) for SpleeterStream.kt; pure C + pffft +
# pthreads, no Android-specific APIs, so the same source builds as a host .dylib/.so for the
# desktop real-model tests (tools/karaoke/build_host_native.sh).
add_library(karaoke_jni SHARED
    karaoke_jni.c
    $<TARGET_OBJECTS:pffft_simd>
    $<TARGET_OBJECTS:pffft_c>)
target_include_directories(karaoke_jni PRIVATE ${SHIM_DIR} ${PW_DIR})
target_compile_options(karaoke_jni PRIVATE ${DSP_FLAGS})
target_link_libraries(karaoke_jni m)
target_link_options(karaoke_jni PRIVATE -Wl,--build-id=none)
target_link_options(karaoke_jni PRIVATE -Wl,-z,max-page-size=16384)
