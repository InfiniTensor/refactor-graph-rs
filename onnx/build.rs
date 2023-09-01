use std::{env, io::Result, path::PathBuf};

fn main() -> Result<()> {
    let project = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let protoc = if env::consts::OS == "windows" {
        "protoc/win64/protoc.exe"
    } else if env::consts::OS == "linux" {
        "protoc/linux-x86_64/protoc"
    } else {
        todo!("Os {} not supported", env::consts::OS);
    };
    std::env::set_var("PROTOC", project.join(protoc));
    prost_build::compile_protos(&["onnx.proto3"], &["onnx-home/onnx/"])?;
    Ok(())
}
