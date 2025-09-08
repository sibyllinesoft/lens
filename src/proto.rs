//! Protocol Buffer definitions for Lens Search
//! 
//! This module contains the generated gRPC service definitions and message types
//! for the Lens Search API with anti-fraud attestation support.

// Include the generated protobuf code
pub mod lens {
    pub mod v1 {
        tonic::include_proto!("lens.v1");
    }
}

// Re-export common types for convenience
pub use lens::v1::*;

// File descriptor set for reflection
pub const FILE_DESCRIPTOR_SET: &[u8] = 
    tonic::include_file_descriptor_set!("lens_search_descriptor");