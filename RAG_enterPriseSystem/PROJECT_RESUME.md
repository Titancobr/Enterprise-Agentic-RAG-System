# Enterprise Agentic RAG System - Resume Writeup

## Project Title

Enterprise Agentic RAG System for Technical Document Intelligence

## One-Line Resume Summary

Built an enterprise-grade Agentic RAG platform using LangGraph, FastAPI, Qdrant, FlashRank, Gemini Embeddings, Portkey, Groq, NeMo Guardrails, Streamlit, Logfire, LangSmith, and RAGAS to answer guarded technical questions over multi-format enterprise documents.

## Resume Bullets

- Built an end-to-end **Enterprise Agentic RAG system** with FastAPI, LangGraph, Qdrant, FlashRank, Gemini Embeddings, Portkey, Groq, NeMo Guardrails, and Streamlit for technical document question answering.

- Designed a multi-stage ingestion pipeline that parses **5 document formats**: PDF, HTML, TXT, DOCX, and PPTX, then chunks, embeds, and indexes content into Qdrant for semantic retrieval.

- Processed **64 source files** and generated **80 indexed document chunks** across true enterprise data and noisy reference data for retrieval testing and evaluation.

- Implemented a LangGraph agent with **planner, retriever, and responder nodes**, allowing the system to route conversational queries directly and technical queries through vector search.

- Integrated **Qdrant vector search** with **FlashRank semantic reranking**, retrieving **15 candidate chunks** and reranking them to the **top 5 most relevant contexts** before response generation.

- Added **NeMo Guardrails** to block off-topic prompts, jailbreak attempts, and prompt-injection style inputs before the request reaches retrieval or generation.

- Built a resilient LLM gateway layer using **Portkey** with Groq model routing, retry handling, fallback configuration, and cache-status detection for production-style LLM reliability.

- Created a RAG evaluation suite with **15 golden Q&A samples**, **6 guardrail test cases**, and **6 evaluation metrics**: faithfulness, answer relevancy, context precision, context recall, answer correctness, and tool correctness.

- Developed a Streamlit chat UI that displays answers, reasoning steps, session memory, and retrieved source chunks to make agent behavior transparent during demos.

- Added observability using **Pydantic Logfire** and **LangSmith tracing** across API requests, guardrails, planning, retrieval, reranking, ingestion, and LLM synthesis.

- Containerized the FastAPI backend with a production Dockerfile and separated production dependencies from local evaluation/demo dependencies.

## Short Project Description

I built an Enterprise Agentic RAG system that answers technical questions over Kubernetes, Intel hardware, and enterprise networking documentation. The system uses a FastAPI backend, LangGraph agent workflow, Qdrant vector database, Gemini embeddings, FlashRank reranking, Portkey LLM gateway, Groq models, NeMo Guardrails, Streamlit UI, Logfire/LangSmith observability, and a RAGAS-based evaluation suite.

The project supports ingestion from PDF, HTML, TXT, DOCX, and PPTX files, processes enterprise and noisy datasets, and evaluates the pipeline with golden question-answer samples, guardrail tests, and multiple RAG quality metrics.

## Strong LinkedIn/GitHub Description

Built a production-style Enterprise Agentic RAG platform for technical document intelligence. The system ingests multi-format enterprise documents, chunks and embeds them, indexes them in Qdrant, retrieves and reranks context with FlashRank, and generates answers through a guarded LangGraph workflow. I added NeMo Guardrails for safety, Portkey for LLM routing/fallback, Logfire and LangSmith for observability, Streamlit apps for chat and evaluation, and a RAGAS-based benchmark suite to measure answer quality.

## Best Version for a Resume Project Section

**Enterprise Agentic RAG System**  
Python, FastAPI, LangGraph, Qdrant, FlashRank, Gemini Embeddings, Portkey, Groq, NeMo Guardrails, Streamlit, RAGAS

- Built a production-style Agentic RAG platform for technical enterprise document Q&A across Kubernetes, Intel hardware, and networking domains.
- Implemented ingestion for **5 file formats** and processed **64 source documents** into **80 indexed chunks** for vector retrieval.
- Designed a LangGraph workflow with planner, retriever, and responder nodes, using Qdrant search plus FlashRank reranking to select the **top 5 contexts** from **15 candidates**.
- Added NeMo Guardrails, Portkey LLM routing/fallback, Logfire/LangSmith observability, Streamlit UI, Docker packaging, and a RAGAS eval suite with **15 golden samples** and **6 metrics**.

## Project Rating

**8/10**

This is a strong portfolio project because it has a real architecture, multiple production-style components, evaluation tooling, observability, safety rails, and deployment packaging. To make it closer to a 10/10 production project, add automated unit/integration tests, CI, stricter error handling, configurable production settings, and documented eval results from actual runs.
