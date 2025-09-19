//! Search command implementation

use anyhow::Result;
use lens_search_engine::{
    parse_full_query, ProgrammingLanguage, QueryBuilder, QueryType, SearchEngine,
};
use std::path::PathBuf;

/// Search the index
pub async fn search_index(
    index_path: PathBuf,
    query: String,
    limit: usize,
    offset: usize,
    fuzzy: bool,
    symbols: bool,
    language: Option<String>,
    file_pattern: Option<String>,
) -> Result<()> {
    // Create search engine
    let search_engine = SearchEngine::new(&index_path).await?;

    let parsed_query =
        parse_full_query(&query).unwrap_or_else(|_| QueryBuilder::new(&query).build());
    let mut query_builder = QueryBuilder::new(&parsed_query.text);

    // Determine query type (CLI flags take precedence over inline tokens)
    if fuzzy {
        query_builder = query_builder.fuzzy();
    } else if symbols {
        query_builder = query_builder.symbol();
    } else {
        query_builder = match parsed_query.query_type {
            QueryType::Exact => query_builder.exact(),
            QueryType::Fuzzy => query_builder.fuzzy(),
            QueryType::Symbol => query_builder.symbol(),
            QueryType::Text => query_builder,
        };
    }

    query_builder = query_builder.limit(limit).offset(offset);

    if let Some(ref lang) = language {
        if let Some(programming_lang) = map_language_string(lang) {
            query_builder = query_builder.language(programming_lang);
        } else {
            eprintln!("Warning: Unknown language '{}', ignoring filter", lang);
        }
    } else if let Some(language) = parsed_query.language_filter.clone() {
        query_builder = query_builder.language(language);
    }

    if let Some(ref pattern) = file_pattern {
        query_builder = query_builder.file_pattern(pattern);
    } else if let Some(pattern) = parsed_query.file_filter.clone() {
        query_builder = query_builder.file_pattern(pattern);
    }

    let search_query = query_builder.build();

    // Execute search
    let results = search_engine.search(&search_query).await?;
    let index_stats = search_engine.get_stats().await.ok();

    let effective_limit = search_query.limit.unwrap_or(limit);
    let effective_offset = search_query.offset.unwrap_or(offset);

    // Display results
    if results.results.is_empty() {
        println!(
            "No results found for '{}' (limit {}, offset {})",
            search_query.text, effective_limit, effective_offset
        );
    } else {
        println!(
            "Found {} results for '{}' (limit {}, offset {}) in {:?}",
            results.total_matches,
            search_query.text,
            effective_limit,
            effective_offset,
            results.search_duration
        );

        if results.from_cache {
            println!("(served from cache)");
        }

        if let Some(stats) = index_stats {
            println!(
                "Index stats: {} docs, size {}, avg doc {:.1}",
                stats.total_documents,
                stats.human_readable_size(),
                stats.average_document_size()
            );
            println!("Supported languages: {}", stats.supported_languages);
        }
        println!();

        for (i, result) in results.results.iter().enumerate() {
            println!(
                "{}. {}:{} (score: {:.2})",
                i + 1,
                result.file_path,
                result.line_number,
                result.score
            );
            println!("   {}", result.content.trim());

            if let Some(ref language) = result.language {
                println!("   language: {}", language.to_string());
            }

            // Show context if available
            if let Some(ref context) = result.context_lines {
                if !context.is_empty() {
                    println!("   Context:");
                    for (j, line) in context.iter().enumerate() {
                        let is_match_line = j == context.len() / 2; // Middle line is the match
                        let prefix = if is_match_line { " → " } else { "   " };
                        println!("{}   {}", prefix, line.trim());
                    }
                }
            }
            println!();
        }
    }

    Ok(())
}

/// Map language string to ProgrammingLanguage enum
fn map_language_string(lang: &str) -> Option<ProgrammingLanguage> {
    match lang.to_lowercase().as_str() {
        "rust" | "rs" => Some(ProgrammingLanguage::Rust),
        "python" | "py" => Some(ProgrammingLanguage::Python),
        "typescript" | "ts" => Some(ProgrammingLanguage::TypeScript),
        "javascript" | "js" => Some(ProgrammingLanguage::JavaScript),
        "go" => Some(ProgrammingLanguage::Go),
        "java" => Some(ProgrammingLanguage::Java),
        "cpp" | "c++" => Some(ProgrammingLanguage::Cpp),
        "c" => Some(ProgrammingLanguage::C),
        _ => None,
    }
}
