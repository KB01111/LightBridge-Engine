fn main() {
    println!("cargo:rerun-if-changed=src/native/canary.cu");
    println!("cargo:rerun-if-env-changed=LIGHTBRIDGE_CUDA_ALLOW_UNSUPPORTED_MSVC");
    if std::env::var_os("CARGO_FEATURE_CUDA_NATIVE").is_none() {
        return;
    }

    let allow_unsupported =
        std::env::var_os("LIGHTBRIDGE_CUDA_ALLOW_UNSUPPORTED_MSVC").is_some_and(|value| value == "1");
    if cfg!(windows) && newest_visual_studio_major().is_some_and(|major| major >= 18) && !allow_unsupported {
        println!(
            "cargo:warning=CUDA native canary rejected: Visual Studio 2026 is outside CUDA 13.1's \
             supported host-compiler range"
        );
        panic!(
            "CUDA native canary rejected: the newest MSVC installation is Visual Studio 2026, \
             outside CUDA 13.1's supported 2019-2022 range. Install/select VS 2022 Build Tools, \
             or explicitly set LIGHTBRIDGE_CUDA_ALLOW_UNSUPPORTED_MSVC=1 for this canary only"
        );
    }

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .cudart("static")
        .flag("-std=c++14")
        .flag("-gencode=arch=compute_89,code=sm_89")
        .flag("-gencode=arch=compute_89,code=compute_89")
        .file("src/native/canary.cu");
    if allow_unsupported {
        build.flag("-allow-unsupported-compiler");
    }
    build.compile("bridge_cuda_canary_v1");
}

fn newest_visual_studio_major() -> Option<u32> {
    let root = std::env::var_os("ProgramFiles(x86)")?;
    let vswhere = std::path::PathBuf::from(root)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    let output = std::process::Command::new(vswhere)
        .args(["-latest", "-products", "*", "-property", "installationVersion"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .split('.')
        .next()?
        .parse()
        .ok()
}
