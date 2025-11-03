use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let sys_rs = manifest_dir.join("src/sys.rs");
    println!("cargo:rerun-if-changed=build.rs");

    // Check tops.rs (existence + empty or invalid content)
    let needs_regen = match fs::read_to_string(&sys_rs) {
        Ok(content) => {
            let lines: Vec<_> = content.lines().collect();
            lines.len() <= 1 || content.trim().is_empty()
        }
        Err(_) => true, // file doesn't exist or can't be read
    };

    // Check sys.rs
    if needs_regen {
        println!(
            "cargo:warning={} not found, running bindgen.sh...",
            sys_rs.display()
        );
        if !run_script_in_dir("bindgen.sh", &manifest_dir) {
            eprintln!(
                "Error: Failed to generate sys.rs. Please install `bindgen` and ensure topsrider is correctly installed."
            );
            std::process::exit(1);
        }
        if sys_rs.exists() {
            if let Err(e) = patch_sys_rs(&sys_rs) {
                eprintln!("Error while patching sys.rs: {e}");
                std::process::exit(1);
            }
        } else {
            eprintln!(
                "Error: sys.rs was not generated. Please install `bindgen` and ensure topsrider is correctly installed."
            );
            std::process::exit(1);
        }
    }
}

/// Run a script inside a specific directory and return success
fn run_script_in_dir(script: &str, dir: &PathBuf) -> bool {
    let script_path = dir.join(script);
    match Command::new("sh")
        .arg(script_path.file_name().unwrap()) // just "bindgen.sh"
        .current_dir(dir) // run inside dir
        .status()
    {
        Ok(status) if status.success() => true,
        _ => false,
    }
}

/// Modify sys.rs: uncomment topsStream_t typedef and insert a `use` line
fn patch_sys_rs(path: &Path) -> io::Result<()> {
    let content = fs::read_to_string(path)?;

    // 1. comment the typedef line
    let commented = content.replace(
        "pub type topsStream_t = *mut itopsStream_t;",
        "// pub type topsStream_t = *mut itopsStream_t;",
    );

    // 2. Add the `use` statement after the first line (if not already there)
    let mut lines: Vec<String> = commented.lines().map(|s| s.to_string()).collect();
    if lines.len() > 1 && !lines[1].contains("use tops_raw::topsStream_t;") {
        lines.insert(1, "use tops_raw::topsStream_t;".to_string());
    }

    fs::write(path, lines.join("\n"))?;
    Ok(())
}
