#!/bin/bash

# Fix grpc test async issues
sed -i 's/    \[test\]/    \[tokio::test\]/g' /home/nathan/Projects/lens/src/grpc/mod.rs
sed -i 's/    fn test_/    async fn test_/g' /home/nathan/Projects/lens/src/grpc/mod.rs
sed -i 's/create_mock_service()/create_mock_service().await/g' /home/nathan/Projects/lens/src/grpc/mod.rs
sed -i 's/create_mock_service_with_failure()/create_mock_service_with_failure().await/g' /home/nathan/Projects/lens/src/grpc/mod.rs
sed -i 's/create_mock_service_with_slow_response(\([^)]*\))/create_mock_service_with_slow_response(\1).await/g' /home/nathan/Projects/lens/src/grpc/mod.rs

echo "Fixed grpc test async issues"