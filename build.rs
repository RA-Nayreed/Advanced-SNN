use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/lif_dense.cu");
    println!("cargo:rerun-if-changed=kernels/event_snn.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    println!("cargo:rustc-env=ADVANCED_SNN_PTX_DIR={}", out_dir.display());

    if Command::new("nvcc").arg("--version").output().is_err() {
        println!("cargo:warning=nvcc was not found; CUDA PTX generation skipped");
        println!("cargo:rustc-env=ADVANCED_SNN_CUDA_AVAILABLE=0");
        return;
    }

    let kernels = [
        ("kernels/lif_dense.cu", "lif_dense.ptx"),
        ("kernels/event_snn.cu", "event_snn.ptx"),
    ];

    for (input, output) in kernels {
        if let Err(error) = compile_ptx(input, &out_dir.join(output)) {
            println!("cargo:warning={error}");
            println!("cargo:rustc-env=ADVANCED_SNN_CUDA_AVAILABLE=0");
            return;
        }
    }

    println!("cargo:rustc-env=ADVANCED_SNN_CUDA_AVAILABLE=1");
}

fn compile_ptx(input: &str, output: &Path) -> Result<(), String> {
    let status = Command::new("nvcc")
        .arg("-ptx")
        .arg("-O3")
        .arg("--std=c++14")
        .arg(input)
        .arg("-o")
        .arg(output)
        .status()
        .map_err(|error| format!("failed to run nvcc for {input}: {error}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("nvcc failed while compiling {input}"))
    }
}
