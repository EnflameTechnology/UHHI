fn main() {
    let paths = vec!["/root/Chopper/tops-software-platform/include/tops"];
    for path in paths {
        println!("cargo:rustc-link-search=native={}", path);
    }

    println!("cargo:rustc-link-lib=dylib=tops");
    println!("cargo:rerun-if-changed=build.rs");

}
