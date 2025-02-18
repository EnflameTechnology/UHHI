fn main() {
    let paths = vec!["/opt/tops/lib"];
    // let paths = vec!["/home/guoqing/caps/build/lib"]; #for link your caps build
    for path in paths {
        println!("cargo:rustc-link-search=native={}", path);
    }

    println!("cargo:rustc-link-lib=dylib=topsrt");
    println!("cargo:rerun-if-changed=build.rs");
}
