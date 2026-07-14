fn main() {
    #[cfg(feature = "ascend")]
    {
        // Locate the Ascend CANN library path
        let ascend_lib_dirs = [
            "/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64",
            "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64",
            "/usr/local/Ascend/ascend-toolkit/latest/lib64",
        ];

        for dir in ascend_lib_dirs {
            println!("cargo:rustc-link-search=native={dir}");
        }

        // Link against the CANN ACL runtime
        println!("cargo:rustc-link-lib=dylib=ascendcl");

        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-env-changed=ASCEND_HOME");
    }
}
