use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set build timestamp for attestation
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    println!("cargo:rustc-env=BUILD_TIMESTAMP={}", timestamp);

    // Generate built.rs for build information
    built::write_built_file().expect("Failed to acquire build-time information");

    // Check if proto files exist before trying to compile them
    let proto_dir = "proto";
    if std::path::Path::new(proto_dir).exists() {
        // Compile protobuf files
        let proto_files: Vec<_> = std::fs::read_dir(proto_dir)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension()? == "proto" {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        if !proto_files.is_empty() {
            let out_dir = env::var("OUT_DIR").unwrap();
            let descriptor_path = std::path::Path::new(&out_dir).join("lens_search_descriptor");
            
            tonic_build::configure()
                .file_descriptor_set_path(&descriptor_path)
                .compile_protos(&proto_files, &[proto_dir])?;
        }
    } else {
        println!("cargo:warning=Proto directory not found, skipping protobuf generation");
    }

    Ok(())
}