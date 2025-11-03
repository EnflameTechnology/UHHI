use std::fs;
use std::io;
use std::io::Write;
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

    // Check tops.rs (existence + empty or invalid content)
    let needs_regen = match fs::read_to_string(&tops_rs) {
        Ok(content) => {
            let lines: Vec<_> = content.lines().collect();
            lines.len() <= 1 || content.trim().is_empty()
        }
        Err(_) => true, // file doesn't exist or can't be read
    };

    // Check tops.rs (check multiple lines inside)
    if needs_regen {
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

        // Fix bug for bindgen which defines size_t incorrectly on some platforms
        if let Err(e) = fix_size_t_typedef(&tops_rs) {
            eprintln!("Warning: Failed to patch tops.rs: {}", e);
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

/// Replace `pub type size_t = ::std::os::raw::c_ulong;` with `pub type size_t = usize;`
fn fix_size_t_typedef(tops_rs: &PathBuf) -> io::Result<()> {
    let content = fs::read_to_string(tops_rs)?;
    let replaced = content.replace(
        "pub type size_t = ::std::os::raw::c_ulong;",
        "pub type size_t = usize;",
    );
    if replaced != content {
        let mut file = fs::File::create(tops_rs)?;
        file.write_all(replaced.as_bytes())?;
    }
    Ok(())
}
