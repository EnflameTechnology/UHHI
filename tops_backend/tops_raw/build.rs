use std::path::PathBuf;
use std::process::Command;

fn main() {
    let paths = vec!["/opt/tops/lib"];
    // let paths = vec!["/home/guoqing/caps/build/lib"]; #for link your caps build
    for path in paths {
        println!("cargo:rustc-link-search=native={}", path);
    }

    println!("cargo:rustc-link-lib=dylib=topsrt");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let tops_rs = manifest_dir.join("src/tops.rs");

    // Check tops.rs
    if !tops_rs.exists() {
        println!("cargo:warning=tops.rs not found, running bindgen.sh...");
        if !run_script_in_dir("bindgen.sh", &manifest_dir) {
            eprintln!(
                "Error: Failed to generate tops.rs. Please install `bindgen` and ensure topsrider is correctly installed."
            );
            std::process::exit(1);
        }
        if !tops_rs.exists() {
            eprintln!(
                "Error: tops.rs was not generated. Please install `bindgen` and ensure topsrider is correctly installed."
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
