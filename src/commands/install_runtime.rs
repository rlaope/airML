//! Install-runtime command implementation.
//!
//! Auto-downloads the ONNX Runtime shared library to `~/.airml/onnxruntime/`
//! and extracts it, writing a config file recording the resolved dylib path.

use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use tar::Archive;

use crate::cli::InstallRuntimeArgs;

/// Execute the install-runtime command.
pub fn execute(args: &InstallRuntimeArgs) -> Result<()> {
    let platform = match &args.platform {
        Some(p) => p.clone(),
        None => detect_platform()?,
    };

    let ort_platform = map_platform(&platform)?;
    let version = &args.version;

    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{version}/onnxruntime-{ort_platform}-{version}.tgz"
    );

    let dest_dir = airml_dir()?.join("onnxruntime").join(format!("v{version}"));

    if dest_dir.exists() && !args.force {
        println!(
            "ONNX Runtime v{version} already present at {}",
            dest_dir.display()
        );
        println!("Use --force to re-download.");
        return Ok(());
    }

    if dest_dir.exists() && args.force {
        eprintln!("--force: removing {}", dest_dir.display());
        fs::remove_dir_all(&dest_dir)
            .with_context(|| format!("Failed to remove existing runtime directory: {}", dest_dir.display()))?;
    }

    fs::create_dir_all(&dest_dir)
        .with_context(|| format!("Failed to create runtime directory: {}", dest_dir.display()))?;

    let tgz_path = dest_dir.join("onnxruntime.tgz");

    eprintln!("Downloading ONNX Runtime v{version} for {platform}...");
    eprintln!("  URL: {url}");

    let bytes = http_download(&url)?;

    eprint!("\r  Downloaded {} bytes", bytes.len());
    eprintln!();

    fs::write(&tgz_path, &bytes)
        .with_context(|| format!("Failed to write archive: {}", tgz_path.display()))?;
    eprintln!("  Saved archive to: {}", tgz_path.display());

    // Extract the tarball into dest_dir.
    eprintln!("  Extracting...");
    extract_tgz(&tgz_path, &dest_dir)
        .with_context(|| format!("Failed to extract {}", tgz_path.display()))?;
    eprintln!("  Extracted to: {}", dest_dir.display());

    // Delete the archive unless --keep-archive was passed.
    if !args.keep_archive {
        fs::remove_file(&tgz_path)
            .with_context(|| format!("Failed to delete archive: {}", tgz_path.display()))?;
    }

    // The tarball extracts to a subdirectory like:
    //   onnxruntime-osx-arm64-1.20.0/lib/libonnxruntime.1.20.0.dylib
    // Find that inner lib/ directory and create a non-versioned symlink.
    let dylib_ext = if platform.starts_with("macos") { "dylib" } else { "so" };
    let versioned_name = format!("libonnxruntime.{version}.{dylib_ext}");
    let symlink_name = format!("libonnxruntime.{dylib_ext}");

    let inner_lib = find_inner_lib(&dest_dir, &versioned_name)
        .with_context(|| {
            format!(
                "Could not find '{versioned_name}' inside {}. \
                 The tarball layout may have changed.",
                dest_dir.display()
            )
        })?;

    // Ensure ~/.airml/onnxruntime/v{version}/lib/ exists.
    let out_lib_dir = dest_dir.join("lib");
    fs::create_dir_all(&out_lib_dir)
        .with_context(|| format!("Failed to create lib dir: {}", out_lib_dir.display()))?;

    let symlink_path = out_lib_dir.join(&symlink_name);

    // Remove any stale symlink/file at the target path.
    if symlink_path.exists() || symlink_path.symlink_metadata().is_ok() {
        fs::remove_file(&symlink_path)
            .with_context(|| format!("Failed to remove stale symlink: {}", symlink_path.display()))?;
    }

    create_symlink(&inner_lib, &symlink_path)
        .with_context(|| {
            format!(
                "Failed to create symlink {} -> {}",
                symlink_path.display(),
                inner_lib.display()
            )
        })?;
    eprintln!("  Symlink: {} -> {}", symlink_path.display(), inner_lib.display());

    // Write ~/.airml/config.toml.
    write_config_toml(&symlink_path, version)?;

    // Print final banner.
    println!();
    println!(
        "Done. Add to your shell:\n\n    export ORT_DYLIB_PATH={path}\n\nOr run airml with the env var inline:\n\n    ORT_DYLIB_PATH={path} airml info -m model.onnx",
        path = symlink_path.display()
    );

    Ok(())
}

/// Detect platform from `std::env::consts`.
fn detect_platform() -> Result<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    let platform = match (os, arch) {
        ("macos", "aarch64") => "macos-arm64",
        ("macos", "x86_64") => "macos-x86_64",
        ("linux", "aarch64") => "linux-arm64",
        ("linux", "x86_64") => "linux-x86_64",
        _ => anyhow::bail!("Unsupported platform: {os}/{arch}. Specify --platform explicitly."),
    };

    Ok(platform.to_owned())
}

/// Map our platform string to the Microsoft release artifact naming convention.
fn map_platform(platform: &str) -> Result<&'static str> {
    match platform {
        "macos-arm64" => Ok("osx-arm64"),
        "macos-x86_64" => Ok("osx-x86_64"),
        "linux-arm64" => Ok("linux-aarch64"),
        "linux-x86_64" => Ok("linux-x64"),
        other => anyhow::bail!(
            "Unknown platform '{other}'. Valid values: macos-arm64, macos-x86_64, linux-arm64, linux-x86_64"
        ),
    }
}

/// Return `~/.airml`, creating it if necessary.
fn airml_dir() -> Result<PathBuf> {
    let dir = dirs::home_dir()
        .context("Cannot determine home directory")?
        .join(".airml");
    fs::create_dir_all(&dir).context("Failed to create ~/.airml")?;
    Ok(dir)
}

/// Decompress and unpack `tgz_path` into `dest_dir`.
fn extract_tgz(tgz_path: &Path, dest_dir: &Path) -> Result<()> {
    let file = fs::File::open(tgz_path)
        .with_context(|| format!("Failed to open archive: {}", tgz_path.display()))?;
    let gz = GzDecoder::new(file);
    let mut archive = Archive::new(gz);
    archive
        .unpack(dest_dir)
        .with_context(|| format!("Failed to unpack archive into {}", dest_dir.display()))?;
    Ok(())
}

/// Walk `base_dir` recursively to find the first file named `target_name`.
///
/// Returns the absolute path to the file, or an error if not found.
fn find_inner_lib(base_dir: &Path, target_name: &str) -> Result<PathBuf> {
    for entry in walkdir(base_dir)? {
        let entry = entry.with_context(|| format!("Error reading entry in {}", base_dir.display()))?;
        if entry.file_name().to_string_lossy() == target_name {
            return Ok(entry.path().to_path_buf());
        }
    }
    anyhow::bail!(
        "File '{}' not found under {}",
        target_name,
        base_dir.display()
    );
}

/// Minimal recursive directory walker that returns `fs::DirEntry`-compatible items.
///
/// We avoid pulling in the `walkdir` crate by using a simple stack-based approach.
fn walkdir(dir: &Path) -> Result<impl Iterator<Item = Result<WalkEntry>>> {
    let mut stack = vec![dir.to_path_buf()];
    let mut entries: Vec<Result<WalkEntry>> = Vec::new();

    while let Some(current) = stack.pop() {
        let read_dir = fs::read_dir(&current)
            .with_context(|| format!("Failed to read directory: {}", current.display()))?;
        for entry in read_dir {
            match entry {
                Ok(e) => {
                    let path = e.path();
                    if path.is_dir() {
                        stack.push(path);
                    } else {
                        entries.push(Ok(WalkEntry { path }));
                    }
                }
                Err(err) => {
                    entries.push(Err(anyhow::Error::from(err)));
                }
            }
        }
    }

    Ok(entries.into_iter())
}

/// Lightweight stand-in for a `walkdir::DirEntry`.
struct WalkEntry {
    path: PathBuf,
}

impl WalkEntry {
    fn file_name(&self) -> &std::ffi::OsStr {
        self.path.file_name().unwrap_or_default()
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

/// Create a symlink from `link_path` pointing to `target`.
///
/// On Unix the link stores the absolute path of `target`.
#[cfg(unix)]
fn create_symlink(target: &Path, link_path: &Path) -> Result<()> {
    std::os::unix::fs::symlink(target, link_path)
        .with_context(|| format!("symlink({}, {})", target.display(), link_path.display()))?;
    Ok(())
}

/// Write `~/.airml/config.toml` recording the runtime dylib path and version.
fn write_config_toml(lib_path: &Path, version: &str) -> Result<()> {
    let config_path = airml_dir()?.join("config.toml");
    // Use a raw string representation safe for TOML (path as a quoted string).
    let path_str = lib_path.display().to_string();
    let content = format!(
        "# airML configuration — auto-generated by `airml install-runtime`\nruntime_path = \"{path_str}\"\nruntime_version = \"{version}\"\n"
    );
    fs::write(&config_path, content)
        .with_context(|| format!("Failed to write config: {}", config_path.display()))?;
    eprintln!("  Config written to: {}", config_path.display());
    Ok(())
}

/// Blocking HTTP GET that returns the full response body.
///
/// ureq 2.x returns `Err` for non-2xx status codes automatically, so no
/// manual status check is required after a successful `call()`.
fn http_download(url: &str) -> Result<Vec<u8>> {
    let response = ureq::get(url)
        .call()
        .with_context(|| format!("HTTP request failed: {url}"))?;

    let mut buf = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut buf)
        .with_context(|| format!("Failed reading response body from {url}"))?;

    Ok(buf)
}
