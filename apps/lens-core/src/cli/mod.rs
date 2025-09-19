//! CLI command modules
//!
//! This module contains the implementations for all CLI subcommands.

pub mod clear;
pub mod index;
pub mod lsp;
pub mod optimize;
pub mod search;
pub mod serve;
pub mod stats;

pub use clear::*;
pub use index::*;
pub use lsp::*;
pub use optimize::*;
pub use search::*;
pub use serve::*;
pub use stats::*;
