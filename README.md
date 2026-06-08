# Insurance RAG System

## Overview

This project implements a domain-specific insurance question answering system based on Retrieval-Augmented Generation (RAG), local LLM inference, and Kubernetes-based deployment. The system retrieves relevant context from an insurance knowledge base and uses a locally served large language model to generate grounded answers.

The project combines three main layers:
- the application layer, which handles the user interface, API access, and retrieval logic,
- the retrieval layer, which uses embeddings and a vector database,
- the inference layer, which performs local LLM inference through `llama.cpp`.

In addition to the functional implementation, the project also includes Kubernetes deployment, persistent storage for the vector database, autoscaling with HPA, and load testing with `k6`.

## Inference Backend Performance Snapshot

- **Model:** Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
- **Backend:** `llama.cpp` + Vulkan
- **GPU:** Radeon RX 580 8GB
- **CPU-only (`-ngl 0`):** 4.7 tokens/sec
- **GPU (`-ngl 40`):** 22.0 tokens/sec
- **GPU (`-ngl 50`):** 22.0 tokens/sec

The inference backend uses `llama.cpp` for local execution of quantized GGUF models. With the Vulkan backend enabled, part of the inference workload is transferred to the AMD GPU, leading to a significant speedup compared to CPU-only execution.

The `-ngl` parameter in `llama.cpp` controls how many model layers are offloaded to the GPU. Experimental testing showed that on the AMD Radeon RX 580, `-ngl 40` provided a clear performance improvement over CPU-only execution, while higher values did not produce any meaningful additional gain. :contentReference[oaicite:0]{index=0}

## Prerequisites

Before running the project, make sure the following are available:

- Python
- Docker
- Minikube
- `kubectl`
- A local `llama.cpp` Vulkan build
- The project source code
- The `k6-test.js` file for load testing

## Running the Local Inference Backend

```bash
cd D:\Users\user\Documents\llama-cpp-vulkan
