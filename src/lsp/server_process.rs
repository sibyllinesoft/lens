//! LSP Server Process Management
//! 
//! Manages language server processes with proper lifecycle and error handling

use anyhow::{anyhow, Result};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::{Child, Command, ChildStdin, ChildStdout};
use tokio::time::timeout;
use tokio::io::AsyncWriteExt;
use tracing::{debug, error, info, warn};

/// Manages an LSP server process
pub struct LspServerProcess {
    child: Child,
    command: String,
    args: Vec<String>,
    timeout_ms: u64,
}

impl LspServerProcess {
    /// Start a new LSP server process
    pub async fn new(command: &str, args: &[&str], timeout_ms: u64) -> Result<Self> {
        info!("Starting LSP server: {} {:?}", command, args);
        
        // Check if command exists
        if !Self::command_exists(command).await {
            return Err(anyhow!("LSP server command not found: {}", command));
        }

        let args_vec: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        
        let mut cmd = Command::new(command);
        cmd.args(&args_vec)
           .stdin(Stdio::piped())
           .stdout(Stdio::piped())
           .stderr(Stdio::piped())
           .kill_on_drop(true);

        let child = cmd.spawn()?;

        debug!("LSP server started with PID: {:?}", child.id());

        Ok(Self {
            child,
            command: command.to_string(),
            args: args_vec,
            timeout_ms,
        })
    }

    /// Check if a command exists in PATH
    async fn command_exists(command: &str) -> bool {
        let output = Command::new("which")
            .arg(command)
            .output()
            .await;
        
        match output {
            Ok(output) => output.status.success(),
            Err(_) => {
                // Fallback: try to run the command with --version
                let fallback = Command::new(command)
                    .arg("--version")
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .await;
                
                fallback.map(|status| status.success()).unwrap_or(false)
            }
        }
    }

    /// Get stdin handle for sending requests
    pub fn stdin(&mut self) -> ChildStdin {
        self.child.stdin.take().expect("Failed to get stdin")
    }

    /// Get stdout handle for reading responses
    pub fn stdout(&mut self) -> ChildStdout {
        self.child.stdout.take().expect("Failed to get stdout")
    }

    /// Check if the process is still running
    pub fn is_running(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(Some(_)) => false, // Process has exited
            Ok(None) => true,     // Process is still running
            Err(_) => false,      // Error checking status, assume dead
        }
    }

    /// Get process ID
    pub fn pid(&self) -> Option<u32> {
        self.child.id()
    }

    /// Wait for process to exit with timeout
    pub async fn wait_with_timeout(&mut self, timeout_ms: u64) -> Result<std::process::ExitStatus> {
        Ok(timeout(Duration::from_millis(timeout_ms), self.child.wait()).await??)
    }

    /// Gracefully shutdown the server
    pub async fn shutdown(&mut self) -> Result<()> {
        if !self.is_running() {
            debug!("LSP server already stopped");
            return Ok(());
        }

        info!("Shutting down LSP server: {} (PID: {:?})", self.command, self.pid());

        // First try graceful shutdown
        if let Some(stdin) = self.child.stdin.as_mut() {
            // Send LSP shutdown sequence
            let shutdown_msg = r#"Content-Length: 56

{"jsonrpc":"2.0","method":"shutdown","id":999999999}"#;
            let exit_msg = r#"Content-Length: 43

{"jsonrpc":"2.0","method":"exit","params":null}"#;
            
            let _ = stdin.write_all(shutdown_msg.as_bytes()).await;
            let _ = stdin.write_all(exit_msg.as_bytes()).await;
            let _ = stdin.flush().await;
        }

        // Wait for graceful exit with timeout
        let graceful_timeout = Duration::from_millis(3000);
        match timeout(graceful_timeout, self.child.wait()).await {
            Ok(Ok(exit_status)) => {
                info!("LSP server exited gracefully: {:?}", exit_status);
                return Ok(());
            }
            Ok(Err(e)) => {
                warn!("Error waiting for graceful exit: {:?}", e);
            }
            Err(_) => {
                warn!("LSP server did not exit gracefully within timeout");
            }
        }

        // Force kill if graceful shutdown failed
        if self.is_running() {
            warn!("Force killing LSP server: {}", self.command);
            match self.child.kill().await {
                Ok(_) => {
                    info!("LSP server force killed");
                    // Wait a bit for the kill to take effect
                    let _ = timeout(Duration::from_millis(1000), self.child.wait()).await;
                }
                Err(e) => {
                    error!("Failed to force kill LSP server: {:?}", e);
                    return Err(anyhow!("Failed to kill LSP server: {:?}", e));
                }
            }
        }

        Ok(())
    }

    /// Restart the server process
    pub async fn restart(&mut self) -> Result<()> {
        info!("Restarting LSP server: {}", self.command);
        
        // Shutdown existing process
        self.shutdown().await?;

        // Start new process
        let mut cmd = Command::new(&self.command);
        cmd.args(&self.args)
           .stdin(Stdio::piped())
           .stdout(Stdio::piped())
           .stderr(Stdio::piped())
           .kill_on_drop(true);

        self.child = cmd.spawn()?;

        info!("LSP server restarted with new PID: {:?}", self.child.id());
        Ok(())
    }

    /// Get server health status
    pub async fn health_check(&mut self) -> ServerHealth {
        if !self.is_running() {
            return ServerHealth {
                is_running: false,
                pid: None,
                uptime_ms: 0,
                memory_usage_kb: 0,
                cpu_usage_percent: 0.0,
            };
        }

        let pid = self.pid();
        let (memory_kb, cpu_percent) = match pid {
            Some(pid) => Self::get_process_stats(pid).await.unwrap_or((0, 0.0)),
            None => (0, 0.0),
        };

        ServerHealth {
            is_running: true,
            pid,
            uptime_ms: 0, // TODO: Track start time
            memory_usage_kb: memory_kb,
            cpu_usage_percent: cpu_percent,
        }
    }

    /// Get process memory and CPU statistics
    async fn get_process_stats(pid: u32) -> Result<(u64, f64)> {
        // Use ps command to get process stats
        let output = Command::new("ps")
            .args(&["-p", &pid.to_string(), "-o", "rss,pcpu", "--no-headers"])
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow!("Failed to get process stats"));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = stdout.trim().split_whitespace().collect();
        
        if parts.len() >= 2 {
            let memory_kb = parts[0].parse::<u64>().unwrap_or(0);
            let cpu_percent = parts[1].parse::<f64>().unwrap_or(0.0);
            Ok((memory_kb, cpu_percent))
        } else {
            Ok((0, 0.0))
        }
    }
}

/// Server health information
#[derive(Debug, Clone)]
pub struct ServerHealth {
    pub is_running: bool,
    pub pid: Option<u32>,
    pub uptime_ms: u64,
    pub memory_usage_kb: u64,
    pub cpu_usage_percent: f64,
}

impl Drop for LspServerProcess {
    fn drop(&mut self) {
        if self.is_running() {
            warn!("LSP server process dropped while still running, killing: {}", self.command);
            // Note: We can't await in Drop, so this is best-effort
            let _ = self.child.start_kill();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_command_exists() {
        // Test with a command that should exist
        assert!(LspServerProcess::command_exists("echo").await);
        
        // Test with a command that shouldn't exist
        assert!(!LspServerProcess::command_exists("definitely_not_a_command_12345").await);
    }

    #[tokio::test]
    async fn test_process_lifecycle() {
        // Use 'cat' as a simple long-running process
        let mut process = LspServerProcess::new("cat", &[], 5000).await.unwrap();
        
        assert!(process.is_running());
        assert!(process.pid().is_some());
        
        // Test health check
        let health = process.health_check().await;
        assert!(health.is_running);
        assert!(health.pid.is_some());
        
        // Test shutdown
        process.shutdown().await.unwrap();
        
        // Give it a moment to fully exit
        sleep(Duration::from_millis(100)).await;
        
        assert!(!process.is_running());
    }
}