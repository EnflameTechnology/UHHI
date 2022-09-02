fn main() {
    let paths = vec!["./tops_raw/lib"];
    for path in paths {
        println!("cargo:rustc-link-search=native={}", path);
    }
    println!("Test......................................................................................");
    println!("cargo:rustc-link-lib=dylib=tops");
    println!("cargo:rerun-if-changed=build.rs");

}
