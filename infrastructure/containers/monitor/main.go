package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/disk"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/shirou/gopsutil/v3/load"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/net"
)

type Monitor struct {
	config          *Config
	attestationData *AttestationData
	metrics         *MetricsCollector
	mu              sync.RWMutex
}

type Config struct {
	MonitorInterval   time.Duration `yaml:"monitor_interval"`
	AttestationMode   bool          `yaml:"attestation_mode"`
	ResultsDir        string        `yaml:"results_dir"`
	PrometheusPort    int           `yaml:"prometheus_port"`
	HTTPPort          int           `yaml:"http_port"`
}

type AttestationData struct {
	SystemID        string                 `json:"system_id"`
	Timestamp       time.Time              `json:"timestamp"`
	Hardware        *HardwareInfo          `json:"hardware"`
	OperatingSystem *OSInfo                `json:"operating_system"`
	Network         *NetworkInfo           `json:"network"`
	Performance     *PerformanceBaseline   `json:"performance"`
	Fingerprint     string                 `json:"fingerprint"`
}

type HardwareInfo struct {
	CPU      *CPUInfo      `json:"cpu"`
	Memory   *MemoryInfo   `json:"memory"`
	Disk     *DiskInfo     `json:"disk"`
	Platform *PlatformInfo `json:"platform"`
}

type CPUInfo struct {
	ModelName    string  `json:"model_name"`
	Cores        int32   `json:"cores"`
	Threads      int32   `json:"threads"`
	MaxFrequency float64 `json:"max_frequency_mhz"`
	CacheSize    int32   `json:"cache_size_kb"`
	Governor     string  `json:"governor"`
}

type MemoryInfo struct {
	TotalGB       float64 `json:"total_gb"`
	Type          string  `json:"type"`
	Speed         string  `json:"speed"`
	AvailableGB   float64 `json:"available_gb"`
}

type DiskInfo struct {
	TotalGB     float64 `json:"total_gb"`
	FreeGB      float64 `json:"free_gb"`
	FileSystem  string  `json:"filesystem"`
	IOPSBaseline int64  `json:"iops_baseline"`
}

type PlatformInfo struct {
	Hostname        string `json:"hostname"`
	Architecture    string `json:"architecture"`
	BootTime        int64  `json:"boot_time"`
	Virtualization  string `json:"virtualization"`
}

type OSInfo struct {
	Name        string `json:"name"`
	Version     string `json:"version"`
	Kernel      string `json:"kernel"`
	KernelArch  string `json:"kernel_arch"`
}

type NetworkInfo struct {
	Interfaces []NetworkInterface `json:"interfaces"`
	Latency    *NetworkLatency    `json:"latency"`
}

type NetworkInterface struct {
	Name      string `json:"name"`
	MTU       int    `json:"mtu"`
	Flags     string `json:"flags"`
	Addresses []string `json:"addresses"`
}

type NetworkLatency struct {
	LocalhostMs float64 `json:"localhost_ms"`
	DNSMs       float64 `json:"dns_ms"`
}

type PerformanceBaseline struct {
	CPUScore    float64 `json:"cpu_score"`
	MemoryScore float64 `json:"memory_score"`
	DiskScore   float64 `json:"disk_score"`
	NetworkScore float64 `json:"network_score"`
}

type MetricsCollector struct {
	cpuUsage      prometheus.Gauge
	memoryUsage   prometheus.Gauge
	diskUsage     prometheus.Gauge
	networkLatency prometheus.Gauge
	queryLatency  prometheus.Histogram
	slaViolations prometheus.Counter
}

func NewMonitor() *Monitor {
	config := &Config{
		MonitorInterval: 1 * time.Second,
		AttestationMode: true,
		ResultsDir:      "/results",
		PrometheusPort:  9090,
		HTTPPort:        8090,
	}
	
	// Override from environment
	if interval := os.Getenv("MONITOR_INTERVAL"); interval != "" {
		if d, err := time.ParseDuration(interval); err == nil {
			config.MonitorInterval = d
		}
	}
	
	if mode := os.Getenv("ATTESTATION_MODE"); mode == "enabled" {
		config.AttestationMode = true
	}
	
	if dir := os.Getenv("RESULTS_DIR"); dir != "" {
		config.ResultsDir = dir
	}
	
	metrics := &MetricsCollector{
		cpuUsage: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "system_cpu_usage_percent",
			Help: "Current CPU usage percentage",
		}),
		memoryUsage: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "system_memory_usage_percent", 
			Help: "Current memory usage percentage",
		}),
		diskUsage: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "system_disk_usage_percent",
			Help: "Current disk usage percentage",
		}),
		networkLatency: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "network_latency_ms",
			Help: "Network latency in milliseconds",
		}),
		queryLatency: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "query_duration_seconds",
			Help:    "Query latency distribution",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 15), // 1ms to ~32s
		}),
		slaViolations: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "sla_violations_total",
			Help: "Total number of SLA violations (>150ms)",
		}),
	}
	
	// Register metrics
	prometheus.MustRegister(
		metrics.cpuUsage,
		metrics.memoryUsage,
		metrics.diskUsage,
		metrics.networkLatency,
		metrics.queryLatency,
		metrics.slaViolations,
	)
	
	return &Monitor{
		config:  config,
		metrics: metrics,
	}
}

func (m *Monitor) CollectAttestationData() error {
	log.Println("Collecting hardware attestation data...")
	
	data := &AttestationData{
		SystemID:  uuid.New().String(),
		Timestamp: time.Now(),
	}
	
	// Hardware information
	data.Hardware = &HardwareInfo{}
	
	// CPU information
	cpuInfo, err := cpu.Info()
	if err == nil && len(cpuInfo) > 0 {
		data.Hardware.CPU = &CPUInfo{
			ModelName:    cpuInfo[0].ModelName,
			Cores:        cpuInfo[0].Cores,
			Threads:      int32(runtime.NumCPU()),
			MaxFrequency: cpuInfo[0].Mhz,
			Governor:     getCPUGovernor(),
		}
	}
	
	// Memory information
	memInfo, err := mem.VirtualMemory()
	if err == nil {
		data.Hardware.Memory = &MemoryInfo{
			TotalGB:     float64(memInfo.Total) / 1024 / 1024 / 1024,
			AvailableGB: float64(memInfo.Available) / 1024 / 1024 / 1024,
			Type:        getMemoryType(),
			Speed:       getMemorySpeed(),
		}
	}
	
	// Disk information
	diskInfo, err := disk.Usage("/")
	if err == nil {
		data.Hardware.Disk = &DiskInfo{
			TotalGB:     float64(diskInfo.Total) / 1024 / 1024 / 1024,
			FreeGB:      float64(diskInfo.Free) / 1024 / 1024 / 1024,
			FileSystem:  diskInfo.Fstype,
			IOPSBaseline: benchmarkDiskIOPS(),
		}
	}
	
	// Platform information
	hostInfo, err := host.Info()
	if err == nil {
		data.Hardware.Platform = &PlatformInfo{
			Hostname:       hostInfo.Hostname,
			Architecture:   hostInfo.KernelArch,
			BootTime:       int64(hostInfo.BootTime),
			Virtualization: hostInfo.VirtualizationSystem,
		}
		
		data.OperatingSystem = &OSInfo{
			Name:       hostInfo.OS,
			Version:    hostInfo.PlatformVersion,
			Kernel:     hostInfo.KernelVersion,
			KernelArch: hostInfo.KernelArch,
		}
	}
	
	// Network information
	interfaces, err := net.Interfaces()
	if err == nil {
		netInfo := &NetworkInfo{
			Interfaces: make([]NetworkInterface, 0, len(interfaces)),
		}
		
		for _, iface := range interfaces {
			if len(iface.Addrs) > 0 {
				addrs := make([]string, len(iface.Addrs))
				for i, addr := range iface.Addrs {
					addrs[i] = addr.Addr
				}
				netInfo.Interfaces = append(netInfo.Interfaces, NetworkInterface{
					Name:      iface.Name,
					MTU:       iface.MTU,
					Flags:     strings.Join(iface.Flags, ","),
					Addresses: addrs,
				})
			}
		}
		
		netInfo.Latency = measureNetworkLatency()
		data.Network = netInfo
	}
	
	// Performance baseline
	data.Performance = &PerformanceBaseline{
		CPUScore:    benchmarkCPU(),
		MemoryScore: benchmarkMemory(),
		DiskScore:   benchmarkDisk(),
		NetworkScore: benchmarkNetwork(),
	}
	
	// Generate fingerprint
	data.Fingerprint = generateFingerprint(data)
	
	m.mu.Lock()
	m.attestationData = data
	m.mu.Unlock()
	
	// Save attestation data
	if err := m.saveAttestationData(data); err != nil {
		return fmt.Errorf("failed to save attestation data: %w", err)
	}
	
	log.Printf("Hardware attestation complete. System ID: %s", data.SystemID)
	return nil
}

func (m *Monitor) saveAttestationData(data *AttestationData) error {
	attestationPath := fmt.Sprintf("%s/attestation-%s.json", m.config.ResultsDir, 
		data.Timestamp.Format("20060102-150405"))
	
	file, err := os.Create(attestationPath)
	if err != nil {
		return err
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(data)
}

func (m *Monitor) StartMetricsCollection(ctx context.Context) {
	ticker := time.NewTicker(m.config.MonitorInterval)
	defer ticker.Stop()
	
	log.Printf("Starting metrics collection every %v", m.config.MonitorInterval)
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.collectMetrics()
		}
	}
}

func (m *Monitor) collectMetrics() {
	// CPU usage
	if cpuPercent, err := cpu.Percent(0, false); err == nil && len(cpuPercent) > 0 {
		m.metrics.cpuUsage.Set(cpuPercent[0])
	}
	
	// Memory usage
	if memInfo, err := mem.VirtualMemory(); err == nil {
		m.metrics.memoryUsage.Set(memInfo.UsedPercent)
	}
	
	// Disk usage
	if diskInfo, err := disk.Usage("/"); err == nil {
		m.metrics.diskUsage.Set(diskInfo.UsedPercent)
	}
	
	// Network latency
	if latency := measureNetworkLatency(); latency != nil {
		m.metrics.networkLatency.Set(latency.LocalhostMs)
	}
}

func (m *Monitor) RecordQueryLatency(duration time.Duration, slaViolation bool) {
	m.metrics.queryLatency.Observe(duration.Seconds())
	if slaViolation {
		m.metrics.slaViolations.Inc()
	}
}

func (m *Monitor) GetSystemStatus() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	status := map[string]interface{}{
		"status": "healthy",
		"timestamp": time.Now(),
		"uptime": time.Since(time.Unix(int64(m.attestationData.Hardware.Platform.BootTime), 0)),
	}
	
	if m.attestationData != nil {
		status["system_id"] = m.attestationData.SystemID
		status["fingerprint"] = m.attestationData.Fingerprint
		status["hardware"] = m.attestationData.Hardware
	}
	
	return status
}

func (m *Monitor) StartHTTPServer() {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery())
	
	// Health endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, m.GetSystemStatus())
	})
	
	// Attestation data endpoint
	router.GET("/attestation", func(c *gin.Context) {
		m.mu.RLock()
		data := m.attestationData
		m.mu.RUnlock()
		
		if data == nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "No attestation data available"})
			return
		}
		
		c.JSON(http.StatusOK, data)
	})
	
	// Prometheus metrics endpoint
	router.GET("/metrics", gin.WrapH(promhttp.Handler()))
	
	// SLA violation recording endpoint
	router.POST("/record-query", func(c *gin.Context) {
		var req struct {
			Duration     float64 `json:"duration_ms"`
			SLAViolation bool    `json:"sla_violation"`
		}
		
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		
		duration := time.Duration(req.Duration * float64(time.Millisecond))
		m.RecordQueryLatency(duration, req.SLAViolation)
		
		c.JSON(http.StatusOK, gin.H{"status": "recorded"})
	})
	
	log.Printf("Starting HTTP server on port %d", m.config.HTTPPort)
	
	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", m.config.HTTPPort),
		Handler: router,
	}
	
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("Failed to start HTTP server: %v", err)
	}
}

// Helper functions for system information gathering

func getCPUGovernor() string {
	out, err := exec.Command("cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

func getMemoryType() string {
	out, err := exec.Command("dmidecode", "-t", "17").Output()
	if err != nil {
		return "unknown"
	}
	// Parse dmidecode output for memory type
	lines := strings.Split(string(out), "\n")
	for _, line := range lines {
		if strings.Contains(line, "Type:") && !strings.Contains(line, "Type Detail:") {
			parts := strings.Split(line, ":")
			if len(parts) > 1 {
				return strings.TrimSpace(parts[1])
			}
		}
	}
	return "unknown"
}

func getMemorySpeed() string {
	out, err := exec.Command("dmidecode", "-t", "17").Output()
	if err != nil {
		return "unknown"
	}
	// Parse dmidecode output for memory speed
	lines := strings.Split(string(out), "\n")
	for _, line := range lines {
		if strings.Contains(line, "Speed:") && !strings.Contains(line, "Configured") {
			parts := strings.Split(line, ":")
			if len(parts) > 1 {
				return strings.TrimSpace(parts[1])
			}
		}
	}
	return "unknown"
}

func measureNetworkLatency() *NetworkLatency {
	start := time.Now()
	_, err := net.Dial("tcp", "127.0.0.1:80")
	localhostMs := float64(time.Since(start).Nanoseconds()) / 1e6
	if err != nil {
		localhostMs = 999.0 // Indicate failure
	}
	
	// DNS latency test
	start = time.Now()
	_, err = net.LookupHost("google.com")
	dnsMs := float64(time.Since(start).Nanoseconds()) / 1e6
	if err != nil {
		dnsMs = 999.0
	}
	
	return &NetworkLatency{
		LocalhostMs: localhostMs,
		DNSMs:       dnsMs,
	}
}

// Benchmark functions
func benchmarkCPU() float64 {
	// Simple CPU benchmark - compute pi
	start := time.Now()
	pi := 0.0
	for i := 0; i < 1000000; i++ {
		pi += 4.0 * (1.0 - 2.0*float64(i%2)) / float64(2*i+1)
	}
	duration := time.Since(start)
	
	// Return operations per second
	return 1000000.0 / duration.Seconds()
}

func benchmarkMemory() float64 {
	// Memory bandwidth test
	size := 100 * 1024 * 1024 // 100MB
	data := make([]byte, size)
	
	start := time.Now()
	for i := 0; i < len(data); i++ {
		data[i] = byte(i % 256)
	}
	duration := time.Since(start)
	
	// Return MB/s
	return float64(size) / 1024 / 1024 / duration.Seconds()
}

func benchmarkDisk() float64 {
	// Disk I/O test
	tempFile := "/tmp/disk-benchmark"
	data := make([]byte, 1024*1024) // 1MB
	
	start := time.Now()
	file, err := os.Create(tempFile)
	if err != nil {
		return 0
	}
	defer func() {
		file.Close()
		os.Remove(tempFile)
	}()
	
	for i := 0; i < 10; i++ {
		_, err := file.Write(data)
		if err != nil {
			return 0
		}
	}
	file.Sync()
	duration := time.Since(start)
	
	// Return MB/s
	return 10.0 / duration.Seconds()
}

func benchmarkDiskIOPS() int64 {
	// Simple IOPS benchmark
	tempFile := "/tmp/iops-benchmark"
	data := make([]byte, 4096) // 4KB blocks
	
	file, err := os.Create(tempFile)
	if err != nil {
		return 0
	}
	defer func() {
		file.Close()
		os.Remove(tempFile)
	}()
	
	start := time.Now()
	operations := 0
	for time.Since(start) < time.Second {
		_, err := file.WriteAt(data, int64(operations%100)*4096)
		if err != nil {
			break
		}
		operations++
	}
	
	return int64(operations)
}

func benchmarkNetwork() float64 {
	// Network benchmark - test localhost throughput
	start := time.Now()
	conn, err := net.Dial("tcp", "127.0.0.1:80")
	if err != nil {
		return 0
	}
	conn.Close()
	
	return 1.0 / time.Since(start).Seconds()
}

func generateFingerprint(data *AttestationData) string {
	// Create a unique fingerprint based on hardware characteristics
	fingerprint := fmt.Sprintf("%s-%s-%d-%d-%.0f",
		data.Hardware.CPU.ModelName,
		data.Hardware.Platform.Architecture,
		data.Hardware.CPU.Cores,
		int64(data.Hardware.Memory.TotalGB),
		data.Hardware.CPU.MaxFrequency,
	)
	
	// Simple hash
	hash := 0
	for _, c := range fingerprint {
		hash = hash*31 + int(c)
	}
	
	return fmt.Sprintf("%x", hash)
}

func main() {
	monitor := NewMonitor()
	
	// Collect attestation data on startup
	if monitor.config.AttestationMode {
		if err := monitor.CollectAttestationData(); err != nil {
			log.Fatalf("Failed to collect attestation data: %v", err)
		}
	}
	
	// Start metrics collection
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	go monitor.StartMetricsCollection(ctx)
	
	// Start HTTP server
	monitor.StartHTTPServer()
}