//! Build script for wavetable-io crate.
//!
//! Compiles the protobuf schema using prost-build.

fn main() {
    // Tell Cargo to rerun this build script if the proto file changes
    println!("cargo:rerun-if-changed=../../../../proto/wavetable.proto");

    // Configure prost-build
    let mut config = prost_build::Config::new();

    // Generate code in OUT_DIR
    config
        .compile_protos(
            &["../../../../proto/wavetable.proto"],
            &["../../../../proto"],
        )
        .expect("Failed to compile protobuf definitions");
}
