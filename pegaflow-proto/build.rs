fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/engine.proto");

    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    let mut prost_config = prost_build::Config::new();
    prost_config.protoc_executable(protoc);

    tonic_prost_build::configure().compile_with_config(
        prost_config,
        &["proto/engine.proto"],
        &["proto"],
    )?;

    Ok(())
}
