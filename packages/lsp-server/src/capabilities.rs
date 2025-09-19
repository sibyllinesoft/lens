//! LSP server capabilities definition
//!
//! This module defines the real capabilities that the Lens LSP server
//! provides to clients, replacing any mock or simulated capabilities.

use lsp_types::*;

/// Create the server capabilities for the Lens LSP server
pub fn create_server_capabilities() -> ServerCapabilities {
    ServerCapabilities {
        // Text document synchronization
        text_document_sync: Some(TextDocumentSyncCapability::Kind(
            TextDocumentSyncKind::INCREMENTAL,
        )),

        // Completion support
        completion_provider: Some(CompletionOptions {
            resolve_provider: Some(false),
            trigger_characters: Some(vec![".".to_string(), ":".to_string(), "_".to_string()]),
            all_commit_characters: None,
            work_done_progress_options: WorkDoneProgressOptions::default(),
            completion_item: None,
        }),

        // Go to definition support
        definition_provider: Some(OneOf::Left(true)),

        // Find references support
        references_provider: Some(OneOf::Left(true)),

        // Workspace symbol search
        workspace_symbol_provider: Some(OneOf::Left(true)),

        // Document symbol provider
        document_symbol_provider: Some(OneOf::Left(true)),

        // Hover support
        hover_provider: Some(HoverProviderCapability::Simple(true)),

        // Execute command support
        execute_command_provider: Some(ExecuteCommandOptions {
            commands: vec![
                "lens.search".to_string(),
                "lens.reindex".to_string(),
                "lens.clearCache".to_string(),
                "lens.showStats".to_string(),
            ],
            work_done_progress_options: WorkDoneProgressOptions::default(),
        }),

        // Workspace capabilities
        workspace: Some(WorkspaceServerCapabilities {
            workspace_folders: Some(WorkspaceFoldersServerCapabilities {
                supported: Some(true),
                change_notifications: Some(OneOf::Left(true)),
            }),
            file_operations: None,
        }),

        // Disable features we don't support yet
        type_definition_provider: None,
        implementation_provider: None,
        document_highlight_provider: None,
        code_action_provider: None,
        code_lens_provider: None,
        document_formatting_provider: None,
        document_range_formatting_provider: None,
        document_on_type_formatting_provider: None,
        rename_provider: None,
        folding_range_provider: None,
        selection_range_provider: None,
        linked_editing_range_provider: None,
        call_hierarchy_provider: None,
        semantic_tokens_provider: None,
        moniker_provider: None,
        inline_value_provider: None,
        inlay_hint_provider: None,
        diagnostic_provider: None,
        experimental: None,

        // Additional required fields in newer LSP versions
        position_encoding: None,
        signature_help_provider: None,
        document_link_provider: None,
        color_provider: None,
        declaration_provider: None,
    }
}

/// Create completion item capabilities
pub fn create_completion_item_capabilities() -> CompletionItemCapability {
    CompletionItemCapability {
        snippet_support: Some(false),
        commit_characters_support: Some(true),
        documentation_format: Some(vec![MarkupKind::PlainText, MarkupKind::Markdown]),
        deprecated_support: Some(true),
        preselect_support: Some(false),
        tag_support: Some(TagSupport {
            value_set: vec![CompletionItemTag::DEPRECATED],
        }),
        insert_replace_support: Some(false),
        resolve_support: None,
        insert_text_mode_support: None,
        label_details_support: Some(false),
    }
}

/// Create hover capabilities
pub fn create_hover_capabilities() -> HoverClientCapabilities {
    HoverClientCapabilities {
        dynamic_registration: Some(false),
        content_format: Some(vec![MarkupKind::PlainText, MarkupKind::Markdown]),
    }
}

/// Create workspace symbol capabilities
pub fn create_workspace_symbol_capabilities() -> WorkspaceSymbolClientCapabilities {
    WorkspaceSymbolClientCapabilities {
        dynamic_registration: Some(false),
        symbol_kind: Some(SymbolKindCapability {
            value_set: Some(vec![
                SymbolKind::FILE,
                SymbolKind::MODULE,
                SymbolKind::NAMESPACE,
                SymbolKind::PACKAGE,
                SymbolKind::CLASS,
                SymbolKind::METHOD,
                SymbolKind::PROPERTY,
                SymbolKind::FIELD,
                SymbolKind::CONSTRUCTOR,
                SymbolKind::ENUM,
                SymbolKind::INTERFACE,
                SymbolKind::FUNCTION,
                SymbolKind::VARIABLE,
                SymbolKind::CONSTANT,
                SymbolKind::STRING,
                SymbolKind::NUMBER,
                SymbolKind::BOOLEAN,
                SymbolKind::ARRAY,
                SymbolKind::OBJECT,
                SymbolKind::KEY,
                SymbolKind::NULL,
                SymbolKind::ENUM_MEMBER,
                SymbolKind::STRUCT,
                SymbolKind::EVENT,
                SymbolKind::OPERATOR,
                SymbolKind::TYPE_PARAMETER,
            ]),
        }),
        tag_support: Some(TagSupport {
            value_set: vec![SymbolTag::DEPRECATED],
        }),
        resolve_support: None,
    }
}

/// Get supported languages for the LSP server
pub fn get_supported_languages() -> Vec<String> {
    vec![
        "rust".to_string(),
        "python".to_string(),
        "typescript".to_string(),
        "javascript".to_string(),
        "go".to_string(),
        "java".to_string(),
        "cpp".to_string(),
        "c".to_string(),
        "ruby".to_string(),
        "php".to_string(),
        "swift".to_string(),
        "kotlin".to_string(),
        "scala".to_string(),
        "clojure".to_string(),
        "elixir".to_string(),
        "markdown".to_string(),
        "plaintext".to_string(),
    ]
}

/// Check if a language is supported
pub fn is_language_supported(language_id: &str) -> bool {
    get_supported_languages().contains(&language_id.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_capabilities() {
        let capabilities = create_server_capabilities();

        // Check that essential capabilities are enabled
        assert!(capabilities.text_document_sync.is_some());
        assert!(capabilities.completion_provider.is_some());
        assert!(capabilities.definition_provider.is_some());
        assert!(capabilities.references_provider.is_some());
        assert!(capabilities.workspace_symbol_provider.is_some());
        assert!(capabilities.execute_command_provider.is_some());
    }

    #[test]
    fn test_execute_commands() {
        let capabilities = create_server_capabilities();

        if let Some(exec_provider) = capabilities.execute_command_provider {
            let commands = exec_provider.commands;
            assert!(commands.contains(&"lens.search".to_string()));
            assert!(commands.contains(&"lens.reindex".to_string()));
        } else {
            panic!("Execute command provider should be enabled");
        }
    }

    #[test]
    fn test_supported_languages() {
        let languages = get_supported_languages();

        assert!(languages.contains(&"rust".to_string()));
        assert!(languages.contains(&"python".to_string()));
        assert!(languages.contains(&"typescript".to_string()));
        assert!(languages.contains(&"javascript".to_string()));

        assert!(is_language_supported("rust"));
        assert!(is_language_supported("python"));
        assert!(!is_language_supported("unsupported"));
    }

    #[test]
    fn test_completion_capabilities() {
        let capabilities = create_server_capabilities();

        if let Some(completion_provider) = capabilities.completion_provider {
            let trigger_chars = completion_provider.trigger_characters.unwrap_or_default();
            assert!(trigger_chars.contains(&".".to_string()));
            assert!(trigger_chars.contains(&":".to_string()));
            assert!(trigger_chars.contains(&"_".to_string()));
        } else {
            panic!("Completion provider should be enabled");
        }
    }
}
