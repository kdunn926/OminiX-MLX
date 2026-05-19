// Compile + link the Swift shim that owns the Core ML model handle.
//
// macOS-only. The shim exposes a tiny C ABI:
//
//   void *coreml_load(const char *mlpackage_path, int compute_units);
//   int   coreml_predict(void *handle,
//                        const float *pixels, int pixel_count,
//                        float *out_buffer,   int out_capacity,
//                        int *out_actual_len);
//   void  coreml_free(void *handle);
//
// `compute_units`:
//   0 = all
//   1 = cpu only
//   2 = cpu + GPU
//   3 = cpu + neural engine
//   4 = neural engine only (macOS 14+)
//
// At build time we compile `src/ane_runner.swift` via swiftc to an
// object file and link it into the Rust crate.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        // Spike is macOS-only; non-mac builds compile a stub.
        return;
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let swift_src = PathBuf::from(&manifest_dir).join("src/ane_runner.swift");
    let obj_out = out_dir.join("ane_runner.o");

    println!("cargo:rerun-if-changed={}", swift_src.display());

    // Compile Swift → object file with C-ABI exports. `-emit-object`
    // produces a .o we can pass to the linker. `-parse-as-library` lets
    // us write @_cdecl functions without a top-level `main`.
    let status = Command::new("swiftc")
        .args([
            "-emit-object",
            "-parse-as-library",
            "-O",
            "-whole-module-optimization",
            "-target", &format!("{}-apple-macosx14.0", env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "arm64".to_string())),
            "-o",
        ])
        .arg(&obj_out)
        .arg(&swift_src)
        .status()
        .expect("failed to invoke swiftc — install Xcode command-line tools");
    if !status.success() {
        panic!("swiftc failed compiling {}", swift_src.display());
    }

    println!("cargo:rustc-link-arg={}", obj_out.display());
    // Frameworks required for Core ML + Foundation + ANE driver bridge.
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    // Swift stdlib + ObjC runtime.
    println!("cargo:rustc-link-lib=dylib=swiftCore");
    println!("cargo:rustc-link-lib=dylib=swiftFoundation");
    println!("cargo:rustc-link-lib=dylib=objc");
    // Path to Swift toolchain's runtime libs.
    let toolchain_lib = "/Library/Developer/CommandLineTools/usr/lib/swift/macosx";
    println!("cargo:rustc-link-search=native={}", toolchain_lib);
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift,-rpath,{}",
        toolchain_lib
    );
}
