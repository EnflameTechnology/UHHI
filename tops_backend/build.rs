fn main() {
    let paths = vec!["/usr/lib"];
    for path in paths {
        println!("cargo:rustc-link-search=native={}", path);
        println!("cargo:rustc-link-arg=-Wl,-rpath={}", path);
    }
    println!("cargo:rustc-link-lib=dylib=topsrt");
    // println!("cargo:rustc-link-lib=dylib=excalibur");
    // println!("cargo:rustc-link-lib=dylib=minst");

}