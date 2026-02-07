#!/usr/bin/env python3
"""Create a mock SQLite database for testing the external database sync functionality.

This script creates a sample database with multiple tables:
- employees: Basic employee information
- projects: Project records
- research_papers: Academic paper metadata
- meeting_notes: Meeting summaries with attendees

Run with: python scripts/create_mock_db.py
"""

import sqlite3
from pathlib import Path

# Create the mock database in the data directory
DB_PATH = Path(__file__).parent.parent / "data" / "mock_company.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def create_database():
    """Create the mock database with sample tables and data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop existing tables if they exist
    cursor.executescript("""
        DROP TABLE IF EXISTS meeting_attendees;
        DROP TABLE IF EXISTS meeting_notes;
        DROP TABLE IF EXISTS project_assignments;
        DROP TABLE IF EXISTS research_papers;
        DROP TABLE IF EXISTS projects;
        DROP TABLE IF EXISTS employees;
        DROP TABLE IF EXISTS departments;
    """)

    # Create departments table
    cursor.execute("""
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create employees table
    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT NOT NULL,
            department_id INTEGER,
            bio TEXT,
            expertise TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        )
    """)

    # Create projects table
    cursor.execute("""
        CREATE TABLE projects (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'active',
            tech_stack TEXT,
            goals TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create project_assignments (many-to-many)
    cursor.execute("""
        CREATE TABLE project_assignments (
            id INTEGER PRIMARY KEY,
            employee_id INTEGER,
            project_id INTEGER,
            role TEXT,
            FOREIGN KEY (employee_id) REFERENCES employees(id),
            FOREIGN KEY (project_id) REFERENCES projects(id)
        )
    """)

    # Create research_papers table
    cursor.execute("""
        CREATE TABLE research_papers (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT,
            authors TEXT,
            keywords TEXT,
            publication_venue TEXT,
            year INTEGER,
            findings TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create meeting_notes table
    cursor.execute("""
        CREATE TABLE meeting_notes (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            date TEXT NOT NULL,
            attendees TEXT,
            agenda TEXT,
            discussion TEXT,
            action_items TEXT,
            decisions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert sample departments
    departments = [
        (1, "Engineering", "Software development and infrastructure"),
        (2, "Research", "AI/ML research and innovation"),
        (3, "Product", "Product management and design"),
        (4, "Data Science", "Analytics and data engineering"),
    ]
    cursor.executemany(
        "INSERT INTO departments (id, name, description) VALUES (?, ?, ?)",
        departments
    )

    # Insert sample employees
    employees = [
        (1, "Alice Chen", "alice@company.com", "Senior ML Engineer", 2,
         "Alice specializes in transformer architectures and has led the development of several production NLP systems.",
         "natural language processing, transformers, deep learning"),
        (2, "Bob Martinez", "bob@company.com", "Staff Engineer", 1,
         "Bob is an expert in distributed systems with 15 years of experience building scalable infrastructure.",
         "distributed systems, Kubernetes, microservices, Go"),
        (3, "Carol Williams", "carol@company.com", "Research Scientist", 2,
         "Carol focuses on reinforcement learning and has published papers on multi-agent systems.",
         "reinforcement learning, multi-agent systems, game theory"),
        (4, "David Kim", "david@company.com", "Product Manager", 3,
         "David leads the AI products team and has shipped three major AI-powered features.",
         "product strategy, AI products, user research"),
        (5, "Eva Johnson", "eva@company.com", "Data Scientist", 4,
         "Eva builds recommendation systems and has expertise in collaborative filtering.",
         "recommendation systems, collaborative filtering, Python"),
        (6, "Frank Lee", "frank@company.com", "ML Platform Engineer", 1,
         "Frank maintains the ML infrastructure including feature stores and model serving.",
         "MLOps, feature stores, model serving, Kubernetes"),
        (7, "Grace Park", "grace@company.com", "Research Engineer", 2,
         "Grace works on efficient inference and model compression techniques.",
         "model compression, quantization, efficient inference"),
        (8, "Henry Zhang", "henry@company.com", "Backend Engineer", 1,
         "Henry designs APIs and database schemas for high-throughput applications.",
         "API design, PostgreSQL, Redis, Python"),
    ]
    cursor.executemany(
        """INSERT INTO employees (id, name, email, role, department_id, bio, expertise) 
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        employees
    )

    # Insert sample projects
    projects = [
        (1, "ArchRAG", 
         "Attributed Community-based Hierarchical Retrieval-Augmented Generation system for enterprise knowledge management",
         "active", "Python, FastMCP, SQLite, OpenAI",
         "Build a production-ready RAG system with hierarchical clustering and community-based retrieval"),
        (2, "Model Serving Platform",
         "Low-latency model serving infrastructure supporting real-time inference at scale",
         "active", "Go, Kubernetes, TensorRT, gRPC",
         "Achieve sub-10ms latency for transformer models with 99.9% availability"),
        (3, "Recommendation Engine",
         "Personalized content recommendation system using collaborative and content-based filtering",
         "active", "Python, TensorFlow, Redis, Kafka",
         "Improve user engagement by 25% through better content personalization"),
        (4, "Knowledge Graph",
         "Enterprise knowledge graph capturing relationships between documents, entities, and concepts",
         "planning", "Neo4j, Python, LLMs",
         "Enable semantic search and automated knowledge discovery across the organization"),
        (5, "AutoML Pipeline",
         "Automated machine learning pipeline for rapid model development and deployment",
         "completed", "Python, Kubeflow, MLflow",
         "Reduce model development time from weeks to days for standard ML tasks"),
    ]
    cursor.executemany(
        """INSERT INTO projects (id, name, description, status, tech_stack, goals) 
           VALUES (?, ?, ?, ?, ?, ?)""",
        projects
    )

    # Insert sample research papers
    papers = [
        (1, "Hierarchical Graph Neural Networks for Document Understanding",
         "We propose a novel hierarchical GNN architecture that captures both local and global document structure. Our method achieves state-of-the-art results on document classification and information extraction tasks by modeling documents as multi-level graphs where nodes represent tokens, sentences, and paragraphs.",
         "Alice Chen, Carol Williams, Grace Park",
         "graph neural networks, document understanding, hierarchical models",
         "NeurIPS 2024", 2024,
         "Our HierGNN achieves 94.2% accuracy on DocVQA, surpassing previous methods by 3.1%. The hierarchical attention mechanism proves crucial for long document understanding."),
        (2, "Efficient Inference for Large Language Models via Dynamic Pruning",
         "We introduce DynaPrune, a dynamic pruning method that adapts model capacity based on input complexity. Unlike static pruning, our approach preserves model quality while achieving 2-4x speedup on real-world inference workloads.",
         "Grace Park, Frank Lee",
         "model compression, dynamic pruning, efficient inference, LLMs",
         "ICML 2024", 2024,
         "DynaPrune reduces inference latency by 65% with less than 1% quality degradation. The method is particularly effective for variable-length inputs."),
        (3, "Multi-Agent Reinforcement Learning for Resource Allocation",
         "We study the problem of distributed resource allocation using multi-agent RL. Our cooperative learning framework enables agents to jointly optimize global objectives while maintaining local decision-making autonomy.",
         "Carol Williams, Bob Martinez",
         "multi-agent RL, resource allocation, cooperative learning",
         "ICLR 2024", 2024,
         "Our MARL approach achieves 23% better resource utilization compared to centralized baselines in data center scheduling scenarios."),
        (4, "Neural Collaborative Filtering with Side Information",
         "We extend neural collaborative filtering to incorporate rich side information including user profiles, item attributes, and contextual signals. Our unified framework seamlessly integrates multiple information sources.",
         "Eva Johnson, Henry Zhang",
         "collaborative filtering, recommendation systems, neural networks",
         "RecSys 2023", 2023,
         "NCF-SI improves recommendation quality by 18% on cold-start items by leveraging item descriptions and user browsing history."),
        (5, "Retrieval-Augmented Generation for Enterprise Search",
         "We present a production RAG system designed for enterprise search use cases. Key innovations include hierarchical document chunking, community-based retrieval, and adaptive answer generation.",
         "Alice Chen, David Kim, Bob Martinez",
         "RAG, enterprise search, information retrieval, LLMs",
         "SIGIR 2024", 2024,
         "Our system reduces answer latency by 40% while improving relevance scores by 22% compared to standard dense retrieval approaches."),
    ]
    cursor.executemany(
        """INSERT INTO research_papers (id, title, abstract, authors, keywords, publication_venue, year, findings) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        papers
    )

    # Insert sample meeting notes
    meetings = [
        (1, "ArchRAG Architecture Review", "2024-01-15",
         "Alice Chen, Bob Martinez, David Kim, Frank Lee",
         "Review proposed architecture for ArchRAG system; Discuss integration points; Plan implementation phases",
         "Alice presented the hierarchical clustering approach based on the ArchRAG paper. Bob raised concerns about scaling the graph construction for large corpora. Frank suggested using incremental updates instead of full rebuilds. David emphasized the need for MCP integration for agent use cases.",
         "Alice: Prototype hierarchical clustering by Jan 22; Bob: Benchmark graph construction on 1M docs; Frank: Design incremental update API",
         "Adopted community-based hierarchy as primary architecture; Will use FastMCP for agent integration; Target 100ms query latency"),
        (2, "ML Platform Q1 Planning", "2024-01-08",
         "Frank Lee, Grace Park, Eva Johnson, Henry Zhang",
         "Review Q4 metrics; Plan Q1 roadmap; Discuss infrastructure improvements",
         "Frank reviewed Q4 metrics: 99.95% uptime, 50ms p99 latency. Grace proposed adding support for quantized models. Eva requested better feature store integration. Henry suggested migrating to new Redis cluster for improved throughput.",
         "Frank: Deploy quantization support by Feb 1; Henry: Complete Redis migration by Jan 31; Eva: Document feature store API",
         "Prioritize quantization support for cost savings; Approved Redis migration budget; Defer real-time feature serving to Q2"),
        (3, "Research Sync - Multi-Agent Systems", "2024-01-22",
         "Carol Williams, Alice Chen, Grace Park",
         "Discuss MARL paper progress; Plan next experiments; Review collaboration opportunities",
         "Carol presented preliminary results on cooperative resource allocation. Alice suggested applying the approach to distributed RAG retrieval. Grace discussed potential for combining with model pruning for efficient multi-agent inference.",
         "Carol: Run scaling experiments to 100 agents; Alice: Draft proposal for multi-agent RAG; Grace: Profile memory usage of multi-agent setup",
         "Submit MARL paper to ICML; Explore multi-agent RAG as future research direction"),
        (4, "Product Roadmap Review", "2024-01-29",
         "David Kim, Alice Chen, Eva Johnson, Bob Martinez",
         "Review product metrics; Prioritize features; Discuss customer feedback",
         "David shared user research findings: customers want better search accuracy and faster responses. Alice proposed integrating ArchRAG for improved search. Eva suggested personalization features based on user history. Bob discussed infrastructure requirements for new features.",
         "David: Create PRD for AI search feature; Alice: Demo ArchRAG to product team; Eva: Design personalization experiment",
         "AI search is top priority for Q2; Will run A/B test for personalization; Need 2 additional engineers for search team"),
    ]
    cursor.executemany(
        """INSERT INTO meeting_notes (id, title, date, attendees, agenda, discussion, action_items, decisions) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        meetings
    )

    # Insert project assignments
    assignments = [
        (1, 1, 1, "Lead"),  # Alice -> ArchRAG
        (2, 2, 1, "Contributor"),  # Bob -> ArchRAG
        (3, 4, 1, "Product Owner"),  # David -> ArchRAG
        (4, 6, 1, "Contributor"),  # Frank -> ArchRAG
        (5, 2, 2, "Lead"),  # Bob -> Model Serving
        (6, 6, 2, "Contributor"),  # Frank -> Model Serving
        (7, 7, 2, "Contributor"),  # Grace -> Model Serving
        (8, 5, 3, "Lead"),  # Eva -> Recommendation
        (9, 8, 3, "Contributor"),  # Henry -> Recommendation
        (10, 1, 4, "Lead"),  # Alice -> Knowledge Graph
        (11, 3, 4, "Contributor"),  # Carol -> Knowledge Graph
    ]
    cursor.executemany(
        """INSERT INTO project_assignments (id, employee_id, project_id, role) 
           VALUES (?, ?, ?, ?)""",
        assignments
    )

    conn.commit()
    conn.close()

    print(f"✅ Created mock database at: {DB_PATH}")
    print("\nTables created:")
    print("  - departments (4 records)")
    print("  - employees (8 records)")
    print("  - projects (5 records)")
    print("  - project_assignments (11 records)")
    print("  - research_papers (5 records)")
    print("  - meeting_notes (4 records)")
    print(f"\nTotal: 37 records across 6 tables")

    return str(DB_PATH)


if __name__ == "__main__":
    db_path = create_database()
    print(f"\nTo test the sync, use connection string:")
    print(f"  sqlite:///{db_path}")
