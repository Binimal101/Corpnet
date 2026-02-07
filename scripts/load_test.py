"""Load testing script for DAC-HRAG."""

from __future__ import annotations

import asyncio
import time
import statistics
from dataclasses import dataclass

import httpx


@dataclass
class LoadTestResult:
    total_requests: int
    successful: int
    failed: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float


async def run_query(client: httpx.AsyncClient, query: str) -> tuple[bool, float]:
    """Run a single query and return (success, latency_ms)."""
    start = time.time()
    try:
        response = await client.post(
            "/query",
            json={"text": query, "top_k": 10},
        )
        latency = (time.time() - start) * 1000
        return response.status_code == 200, latency
    except Exception:
        return False, (time.time() - start) * 1000


async def load_test(
    base_url: str = "http://localhost:8000",
    num_requests: int = 100,
    concurrency: int = 10,
) -> LoadTestResult:
    """Run a load test against the API."""
    
    queries = [
        "How does the ingestion pipeline work?",
        "What is the clustering algorithm?",
        "Explain the routing mechanism",
        "What are the main components?",
        "How is the hierarchy built?",
    ]
    
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_query(i: int) -> tuple[bool, float]:
            async with semaphore:
                query = queries[i % len(queries)]
                return await run_query(client, query)
        
        start = time.time()
        results = await asyncio.gather(*[limited_query(i) for i in range(num_requests)])
        total_time = time.time() - start
    
    successes = [r for r in results if r[0]]
    latencies = [r[1] for r in results]
    
    latencies.sort()
    
    return LoadTestResult(
        total_requests=num_requests,
        successful=len(successes),
        failed=num_requests - len(successes),
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=latencies[int(len(latencies) * 0.5)],
        p95_latency_ms=latencies[int(len(latencies) * 0.95)],
        p99_latency_ms=latencies[int(len(latencies) * 0.99)],
        requests_per_second=num_requests / total_time,
    )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Load test DAC-HRAG")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    
    args = parser.parse_args()
    
    print(f"Running load test: {args.requests} requests, {args.concurrency} concurrent")
    print(f"Target: {args.url}")
    print()
    
    result = asyncio.run(load_test(args.url, args.requests, args.concurrency))
    
    print(f"Total requests:     {result.total_requests}")
    print(f"Successful:         {result.successful}")
    print(f"Failed:             {result.failed}")
    print(f"Avg latency:        {result.avg_latency_ms:.1f} ms")
    print(f"P50 latency:        {result.p50_latency_ms:.1f} ms")
    print(f"P95 latency:        {result.p95_latency_ms:.1f} ms")
    print(f"P99 latency:        {result.p99_latency_ms:.1f} ms")
    print(f"Requests/sec:       {result.requests_per_second:.1f}")


if __name__ == "__main__":
    main()
