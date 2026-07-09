use std::path::PathBuf;

use anyhow::{anyhow, bail, Result};

pub const CUDA_UNAVAILABLE: &str = "CUDA support is unavailable";

pub fn compiled_ptx_available() -> bool {
    option_env!("ADVANCED_SNN_CUDA_AVAILABLE") == Some("1")
}

pub fn ptx_path(file_name: &str) -> Result<PathBuf> {
    if !compiled_ptx_available() {
        bail!("{CUDA_UNAVAILABLE}: nvcc did not compile CUDA PTX during the Cargo build");
    }

    let dir = option_env!("ADVANCED_SNN_PTX_DIR")
        .ok_or_else(|| anyhow!("{CUDA_UNAVAILABLE}: PTX output directory was not recorded"))?;
    let path = PathBuf::from(dir).join(file_name);
    if !path.exists() {
        bail!("{CUDA_UNAVAILABLE}: PTX file {} was not found", path.display());
    }
    Ok(path)
}

pub fn unavailable(reason: impl AsRef<str>) -> anyhow::Error {
    anyhow!("{CUDA_UNAVAILABLE}: {}", reason.as_ref())
}

pub fn is_cuda_unavailable(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.to_string().contains(CUDA_UNAVAILABLE))
}
