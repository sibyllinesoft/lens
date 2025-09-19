//! Common types and utilities shared across lens packages
//!
//! This crate provides shared types, enums, and utility functions
//! that are used across multiple lens packages to avoid duplication.

pub mod error;
pub mod language;
pub mod results;

// Re-export common types for convenience
pub use error::LensError;
pub use language::{ParsedContent, ProgrammingLanguage};
pub use results::{IndexStats, SearchResult, SearchResultType, SearchResults};
