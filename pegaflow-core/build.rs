fn main() {
    #[cfg(feature = "ascend")]
    {
        // Resolve Ascend CANN library paths dynamically from environment.
        // Priority: ASCEND_HOME_PATH > ASCEND_HOME > hardcoded fallbacks.
        let ascend_home = std::env::var("ASCEND_HOME_PATH")
            .or_else(|_| std::env::var("ASCEND_HOME"))
            .ok();

        let mut lib_dirs: Vec<String> = Vec::new();
        if let Some(ref home) = ascend_home {
            lib_dirs.push(format!("{home}/aarch64-linux/lib64"));
            lib_dirs.push(format!("{home}/lib64"));
        }

        let fallbacks = [
            "/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64",
            "/usr/local/Ascend/cann-8.5.1/lib64",
            "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64",
            "/usr/local/Ascend/ascend-toolkit/latest/lib64",
            "/usr/local/Ascend/cann/aarch64-linux/lib64",
            "/usr/local/Ascend/cann/lib64",
        ];
        for fb in fallbacks {
            if !lib_dirs.contains(&fb.to_string()) {
                lib_dirs.push(fb.to_string());
            }
        }

        for dir in &lib_dirs {
            println!("cargo:rustc-link-search=native={dir}");
        }

        println!("cargo:rustc-link-lib=dylib=ascendcl");

        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-env-changed=ASCEND_HOME_PATH");
        println!("cargo:rerun-if-env-changed=ASCEND_HOME");
    }
}
