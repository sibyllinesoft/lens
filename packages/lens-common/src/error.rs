//! Common error types
//!
//! Shared error handling across lens packages

use serde::{Deserialize, Serialize};
use std::fmt;

/// Common error type for lens operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LensError {
    /// Index operation errors
    IndexError {
        message: String,
        details: Option<String>,
    },
    /// Search operation errors
    SearchError {
        message: String,
        query: Option<String>,
    },
    /// Language parsing errors
    LanguageError {
        message: String,
        language: Option<String>,
        file_path: Option<String>,
    },
    /// LSP protocol errors
    LspError {
        message: String,
        method: Option<String>,
    },
    /// IO-related errors
    IoError {
        message: String,
        path: Option<String>,
    },
    /// Configuration errors
    ConfigError {
        message: String,
        field: Option<String>,
    },
    /// Generic errors
    Generic { message: String },
}

impl LensError {
    /// Create a new index error
    pub fn index<S: Into<String>>(message: S) -> Self {
        Self::IndexError {
            message: message.into(),
            details: None,
        }
    }

    /// Create a new index error with details
    pub fn index_with_details<S: Into<String>, D: Into<String>>(message: S, details: D) -> Self {
        Self::IndexError {
            message: message.into(),
            details: Some(details.into()),
        }
    }

    /// Create a new search error
    pub fn search<S: Into<String>>(message: S) -> Self {
        Self::SearchError {
            message: message.into(),
            query: None,
        }
    }

    /// Create a new search error with query
    pub fn search_with_query<S: Into<String>, Q: Into<String>>(message: S, query: Q) -> Self {
        Self::SearchError {
            message: message.into(),
            query: Some(query.into()),
        }
    }

    /// Create a new language error
    pub fn language<S: Into<String>>(message: S) -> Self {
        Self::LanguageError {
            message: message.into(),
            language: None,
            file_path: None,
        }
    }

    /// Create a new language error with context
    pub fn language_with_context<S: Into<String>, L: Into<String>, P: Into<String>>(
        message: S,
        language: L,
        file_path: P,
    ) -> Self {
        Self::LanguageError {
            message: message.into(),
            language: Some(language.into()),
            file_path: Some(file_path.into()),
        }
    }

    /// Create a new LSP error
    pub fn lsp<S: Into<String>>(message: S) -> Self {
        Self::LspError {
            message: message.into(),
            method: None,
        }
    }

    /// Create a new LSP error with method
    pub fn lsp_with_method<S: Into<String>, M: Into<String>>(message: S, method: M) -> Self {
        Self::LspError {
            message: message.into(),
            method: Some(method.into()),
        }
    }

    /// Create a new IO error
    pub fn io<S: Into<String>>(message: S) -> Self {
        Self::IoError {
            message: message.into(),
            path: None,
        }
    }

    /// Create a new IO error with path
    pub fn io_with_path<S: Into<String>, P: Into<String>>(message: S, path: P) -> Self {
        Self::IoError {
            message: message.into(),
            path: Some(path.into()),
        }
    }

    /// Create a new config error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::ConfigError {
            message: message.into(),
            field: None,
        }
    }

    /// Create a new config error with field
    pub fn config_with_field<S: Into<String>, F: Into<String>>(message: S, field: F) -> Self {
        Self::ConfigError {
            message: message.into(),
            field: Some(field.into()),
        }
    }

    /// Create a new generic error
    pub fn generic<S: Into<String>>(message: S) -> Self {
        Self::Generic {
            message: message.into(),
        }
    }

    /// Get the error message
    pub fn message(&self) -> &str {
        match self {
            Self::IndexError { message, .. } => message,
            Self::SearchError { message, .. } => message,
            Self::LanguageError { message, .. } => message,
            Self::LspError { message, .. } => message,
            Self::IoError { message, .. } => message,
            Self::ConfigError { message, .. } => message,
            Self::Generic { message } => message,
        }
    }
}

impl fmt::Display for LensError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexError { message, details } => {
                write!(f, "Index error: {}", message)?;
                if let Some(details) = details {
                    write!(f, " ({})", details)?;
                }
                Ok(())
            }
            Self::SearchError { message, query } => {
                write!(f, "Search error: {}", message)?;
                if let Some(query) = query {
                    write!(f, " (query: '{}')", query)?;
                }
                Ok(())
            }
            Self::LanguageError {
                message,
                language,
                file_path,
            } => {
                write!(f, "Language error: {}", message)?;
                if let Some(language) = language {
                    write!(f, " (language: {})", language)?;
                }
                if let Some(file_path) = file_path {
                    write!(f, " (file: {})", file_path)?;
                }
                Ok(())
            }
            Self::LspError { message, method } => {
                write!(f, "LSP error: {}", message)?;
                if let Some(method) = method {
                    write!(f, " (method: {})", method)?;
                }
                Ok(())
            }
            Self::IoError { message, path } => {
                write!(f, "IO error: {}", message)?;
                if let Some(path) = path {
                    write!(f, " (path: {})", path)?;
                }
                Ok(())
            }
            Self::ConfigError { message, field } => {
                write!(f, "Config error: {}", message)?;
                if let Some(field) = field {
                    write!(f, " (field: {})", field)?;
                }
                Ok(())
            }
            Self::Generic { message } => write!(f, "{}", message),
        }
    }
}

impl std::error::Error for LensError {}

/// Result type using LensError
pub type LensResult<T> = Result<T, LensError>;

/// Convert from anyhow::Error to LensError
impl From<anyhow::Error> for LensError {
    fn from(err: anyhow::Error) -> Self {
        Self::generic(err.to_string())
    }
}

/// Convert from std::io::Error to LensError
impl From<std::io::Error> for LensError {
    fn from(err: std::io::Error) -> Self {
        Self::io(err.to_string())
    }
}
