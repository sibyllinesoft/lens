// build.rs - Build configuration for Rust migration
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate build info for attestation
    built::write_built_file().expect("Failed to acquire build-time information");

    // Compile protobuf files with enhanced configuration
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .file_descriptor_set_path(out_dir.join("lens_search_descriptor.bin"))
        .compile_protos(&["proto/lens.proto"], &["proto"])?;

    // Ensure we're in 'real' mode, never 'mock'
    println!("cargo:rustc-env=LENS_MODE=real");
    
    // Capture additional build metadata
    println!("cargo:rustc-env=BUILD_TIMESTAMP={}", env::var("SOURCE_DATE_EPOCH")
        .unwrap_or_else(|_| chrono::Utc::now().timestamp().to_string()));

    // Tell cargo to invalidate the built crate whenever files change
    println!("cargo:rerun-if-changed=proto/lens.proto");
    println!("cargo:rerun-if-changed=build.rs");

    Ok(())
}