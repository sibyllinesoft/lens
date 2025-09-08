use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;

/// Ground truth data for benchmark validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruth {
    pub queries: Vec<GroundTruthItem>,
    pub metadata: GroundTruthMetadata,
}

/// Individual ground truth item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthItem {
    pub query: String,
    pub query_type: QueryType,
    pub expected_results: Vec<ExpectedResult>,
    pub success_criteria: SuccessCriteria,
    pub language: Option<String>,
    pub slice: String,
}

/// Type of benchmark query
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryType {
    ExactMatch,
    Identifier,
    Structural,
    Semantic,
    Hybrid,
}

/// Expected result for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedResult {
    pub file_path: String,
    pub line_number: Option<u32>,
    pub relevance_score: f64,
    pub snippet: Option<String>,
}

/// Success criteria for ground truth validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    pub min_results: usize,
    pub required_files: Vec<String>,
    pub max_latency_ms: u64,
    pub min_relevance_score: f64,
}

/// Metadata for ground truth dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthMetadata {
    pub version: String,
    pub created_at: String,
    pub corpus_fingerprint: String,
    pub total_queries: usize,
    pub query_distribution: HashMap<String, usize>,
}

/// Ground truth loader and validator
pub struct GroundTruthLoader {
    data_path: String,
    cached_ground_truth: Option<GroundTruth>,
}

impl GroundTruthLoader {
    /// Create new ground truth loader
    pub fn new(data_path: String) -> Self {
        Self {
            data_path,
            cached_ground_truth: None,
        }
    }

    /// Load ground truth data from disk
    pub async fn load(&mut self) -> Result<&GroundTruth> {
        if self.cached_ground_truth.is_none() {
            let content = tokio::fs::read_to_string(&self.data_path).await?;
            let ground_truth: GroundTruth = serde_json::from_str(&content)?;
            self.cached_ground_truth = Some(ground_truth);
        }
        Ok(self.cached_ground_truth.as_ref().unwrap())
    }

    /// Get queries by slice
    pub async fn get_queries_by_slice(&mut self, slice: &str) -> Result<Vec<GroundTruthItem>> {
        let ground_truth = self.load().await?;
        Ok(ground_truth
            .queries
            .iter()
            .filter(|item| item.slice == slice)
            .cloned()
            .collect())
    }

    /// Get queries by type
    pub async fn get_queries_by_type(&mut self, query_type: QueryType) -> Result<Vec<GroundTruthItem>> {
        let ground_truth = self.load().await?;
        Ok(ground_truth
            .queries
            .iter()
            .filter(|item| matches!(&item.query_type, query_type))
            .cloned()
            .collect())
    }

    /// Validate ground truth against corpus
    pub async fn validate_against_corpus(&mut self, corpus_files: &[String]) -> Result<ValidationReport> {
        let ground_truth = self.load().await?;
        let mut report = ValidationReport {
            total_queries: ground_truth.queries.len(),
            valid_queries: 0,
            invalid_queries: Vec::new(),
            missing_files: Vec::new(),
        };

        for item in &ground_truth.queries {
            let mut valid = true;
            
            // Check if expected files exist in corpus
            for expected in &item.expected_results {
                if !corpus_files.contains(&expected.file_path) {
                    report.missing_files.push(expected.file_path.clone());
                    valid = false;
                }
            }
            
            // Check required files
            for required_file in &item.success_criteria.required_files {
                if !corpus_files.contains(required_file) {
                    report.missing_files.push(required_file.clone());
                    valid = false;
                }
            }

            if valid {
                report.valid_queries += 1;
            } else {
                report.invalid_queries.push(item.query.clone());
            }
        }

        Ok(report)
    }
}

/// Validation report for ground truth data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub total_queries: usize,
    pub valid_queries: usize,
    pub invalid_queries: Vec<String>,
    pub missing_files: Vec<String>,
}

impl ValidationReport {
    /// Calculate pass rate
    pub fn pass_rate(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            (self.valid_queries as f64 / self.total_queries as f64) * 100.0
        }
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.pass_rate() >= 90.0 // 90% threshold for validation
    }
}

/// Ground truth generation utilities
pub struct GroundTruthGenerator;

impl GroundTruthGenerator {
    /// Generate smoke test ground truth from corpus
    pub async fn generate_smoke_dataset(corpus_files: &[String]) -> Result<GroundTruth> {
        let mut queries = Vec::new();
        let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();

        // Generate stratified sample of queries
        queries.extend(Self::generate_identifier_queries(corpus_files)?);
        queries.extend(Self::generate_structural_queries(corpus_files)?);
        queries.extend(Self::generate_semantic_queries(corpus_files)?);

        let metadata = GroundTruthMetadata {
            version: format!("smoke-{}", timestamp),
            created_at: timestamp,
            corpus_fingerprint: Self::calculate_corpus_fingerprint(corpus_files),
            total_queries: queries.len(),
            query_distribution: Self::calculate_query_distribution(&queries),
        };

        Ok(GroundTruth {
            queries,
            metadata,
        })
    }

    fn generate_identifier_queries(corpus_files: &[String]) -> Result<Vec<GroundTruthItem>> {
        // Simple implementation - would be more sophisticated in practice
        let mut queries = Vec::new();
        
        // Sample query patterns
        let patterns = vec!["function", "class", "interface", "struct", "async", "const"];
        
        for pattern in patterns {
            queries.push(GroundTruthItem {
                query: pattern.to_string(),
                query_type: QueryType::Identifier,
                expected_results: vec![], // Would populate from actual analysis
                success_criteria: SuccessCriteria {
                    min_results: 1,
                    required_files: vec![],
                    max_latency_ms: 2000,
                    min_relevance_score: 0.5,
                },
                language: None,
                slice: "SMOKE_DEFAULT".to_string(),
            });
        }
        
        Ok(queries)
    }

    fn generate_structural_queries(_corpus_files: &[String]) -> Result<Vec<GroundTruthItem>> {
        let mut queries = Vec::new();
        
        queries.push(GroundTruthItem {
            query: "class * extends".to_string(),
            query_type: QueryType::Structural,
            expected_results: vec![],
            success_criteria: SuccessCriteria {
                min_results: 1,
                required_files: vec![],
                max_latency_ms: 2000,
                min_relevance_score: 0.6,
            },
            language: Some("typescript".to_string()),
            slice: "SMOKE_DEFAULT".to_string(),
        });
        
        Ok(queries)
    }

    fn generate_semantic_queries(_corpus_files: &[String]) -> Result<Vec<GroundTruthItem>> {
        let mut queries = Vec::new();
        
        queries.push(GroundTruthItem {
            query: "authentication logic".to_string(),
            query_type: QueryType::Semantic,
            expected_results: vec![],
            success_criteria: SuccessCriteria {
                min_results: 1,
                required_files: vec![],
                max_latency_ms: 3000,
                min_relevance_score: 0.4,
            },
            language: None,
            slice: "SMOKE_DEFAULT".to_string(),
        });
        
        Ok(queries)
    }

    fn calculate_corpus_fingerprint(corpus_files: &[String]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        let mut sorted_files = corpus_files.to_vec();
        sorted_files.sort();
        
        for file in &sorted_files {
            file.hash(&mut hasher);
        }
        
        format!("{:x}", hasher.finish())
    }

    fn calculate_query_distribution(queries: &[GroundTruthItem]) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        
        for query in queries {
            let key = format!("{:?}", query.query_type);
            *distribution.entry(key).or_insert(0) += 1;
        }
        
        distribution
    }
}