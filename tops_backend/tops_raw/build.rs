fn main() {
    let paths = vec!["./lib", "/lib/x86_64-linux-gnu/"];
    for path in paths {
        println!("cargo:rustc-link-search=native={}", path);
    }
    
    println!("cargo:rustc-link-lib=dylib=tops_api64");
    println!("cargo:rerun-if-changed=build.rs");

}
