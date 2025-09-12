# Lens Search Engine - TypeScript Baseline Performance Report

**Timestamp**: 2025-09-09T05:30:28.843Z
**System**: TypeScript (Baseline)
**Server**: http://localhost:3000

## Individual Query Performance

| Query | P95 Latency | Avg Latency | Success Rate | Requests |
|-------|-------------|-------------|--------------|----------|
| function | 5.87ms | 3.05ms | 100.0% | 50 |
| class | 4.53ms | 2.45ms | 100.0% | 50 |
| user | 3.84ms | 2.28ms | 100.0% | 50 |
| async | 5.09ms | 2.44ms | 100.0% | 50 |
| getUserById | 4.12ms | 2.35ms | 100.0% | 50 |
| UserService | 5.07ms | 2.29ms | 100.0% | 50 |
| authentication | 3.83ms | 2.38ms | 100.0% | 50 |
| database connection | 3.56ms | 2.06ms | 100.0% | 50 |

## Overall Performance Summary

- **Average Latency**: 2.41ms
- **Average P95 Latency**: 4.49ms
- **Queries Successfully Tested**: 8

## Concurrent Load Test Results

- **Concurrent Users**: 20
- **Total Requests**: 200
- **Success Rate**: 100.0%
- **Queries per Second**: 1727.16
- **Load Test P95 Latency**: 14.32ms
- **Load Test Average Latency**: 6.92ms

## System Characteristics (TypeScript Baseline)

- **Runtime**: Node.js with TypeScript
- **Server Framework**: Express/Fastify
- **Search Engine**: JavaScript implementation
- **Memory Management**: V8 garbage collector
- **Concurrency**: Event loop based
