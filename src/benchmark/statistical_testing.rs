//! Statistical testing infrastructure
//! Minimal implementation for test compilation

use serde::{Deserialize, Serialize};
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestConfig {
    pub significance_level: f64,
    pub power: f64,
    pub effect_size: f64,
    pub sample_size: usize,
}

impl Default for StatisticalTestConfig {
    fn default() -> Self {
        Self {
            significance_level: 0.05,
            power: 0.8,
            effect_size: 0.5,
            sample_size: 100,
        }
    }
}

pub struct StatisticalTestRunner {
    config: StatisticalTestConfig,
}

impl StatisticalTestRunner {
    pub fn new(config: StatisticalTestConfig) -> Self {
        Self { config }
    }

    pub async fn run_significance_tests(&self) -> Result<StatisticalTestResult> {
        // Minimal implementation for compilation
        Ok(StatisticalTestResult {
            test_name: "default".to_string(),
            p_value: 0.01,
            significant: true,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestResult {
    pub test_name: String,
    pub p_value: f64,
    pub significant: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistical_test_config_default() {
        let config = StatisticalTestConfig::default();
        assert_eq!(config.significance_level, 0.05);
        assert_eq!(config.power, 0.8);
        assert_eq!(config.effect_size, 0.5);
        assert_eq!(config.sample_size, 100);
    }

    #[test]
    fn test_statistical_test_runner_creation() {
        let config = StatisticalTestConfig::default();
        let runner = StatisticalTestRunner::new(config);
        assert_eq!(runner.config.significance_level, 0.05);
    }

    #[tokio::test]
    async fn test_significance_tests() {
        let config = StatisticalTestConfig::default();
        let runner = StatisticalTestRunner::new(config);
        
        let result = runner.run_significance_tests().await;
        assert!(result.is_ok());
        
        let test_result = result.unwrap();
        assert_eq!(test_result.test_name, "default");
        assert_eq!(test_result.p_value, 0.01);
        assert!(test_result.significant);
    }

    #[test]
    fn test_statistical_test_result_creation() {
        let result = StatisticalTestResult {
            test_name: "t-test".to_string(),
            p_value: 0.03,
            significant: true,
        };
        
        assert_eq!(result.test_name, "t-test");
        assert_eq!(result.p_value, 0.03);
        assert!(result.significant);
    }

    #[test]
    fn test_custom_statistical_config() {
        let config = StatisticalTestConfig {
            significance_level: 0.01,
            power: 0.9,
            effect_size: 0.8,
            sample_size: 200,
        };
        
        assert_eq!(config.significance_level, 0.01);
        assert_eq!(config.power, 0.9);
        assert_eq!(config.effect_size, 0.8);
        assert_eq!(config.sample_size, 200);
    }

    #[test]
    fn test_non_significant_result() {
        let result = StatisticalTestResult {
            test_name: "chi-square".to_string(),
            p_value: 0.12,
            significant: false,
        };
        
        assert_eq!(result.test_name, "chi-square");
        assert_eq!(result.p_value, 0.12);
        assert!(!result.significant);
    }
}