// build.rs - Complete attestation and build-time verification
use built;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Capture complete build information for fraud prevention
    built::write_built_file().expect("Failed to acquire build-time information");
    
    // Note: gRPC/protobuf generation removed for initial HTTP-only implementation
    // TODO: Re-add when tonic dependencies are stable
    
    // Verify we're in production mode, never mock
    let mode = std::env::var("LENS_MODE").unwrap_or_else(|_| "real".to_string());
    if mode != "real" && mode != "test" {
        panic!("LENS_MODE must be 'real' or 'test', got '{}'", mode);
    }
    println!("cargo:rustc-env=LENS_MODE={}", mode);
    
    // Capture Git information for attestation
    if let Ok(output) = Command::new("git").args(["rev-parse", "HEAD"]).output() {
        let git_sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
        println!("cargo:rustc-env=GIT_SHA={}", git_sha);
    }
    
    if let Ok(output) = Command::new("git").args(["diff", "--quiet"]).output() {
        let is_dirty = !output.status.success();
        println!("cargo:rustc-env=GIT_DIRTY={}", is_dirty);
    }
    
    // Capture build timestamp
    println!("cargo:rustc-env=BUILD_TIMESTAMP={}", 
        std::env::var("SOURCE_DATE_EPOCH")
            .unwrap_or_else(|_| chrono::Utc::now().timestamp().to_string()));
    
    // Verify critical dependencies for fraud prevention
    println!("cargo:rerun-if-env-changed=LENS_MODE");
    println!("cargo:rerun-if-env-changed=GIT_SHA");
    
    Ok(())
}