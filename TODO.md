# Dirghayu TODO - Data Schemas & API Specifications

## Overview

This document outlines the implementation plan for data schemas and API specifications for the Dirghayu genomics platform. Based on comprehensive analysis, we recommend a hybrid approach combining multiple technologies for optimal performance and flexibility.

## Data Schema Approaches Analysis

### 1. TOML Schema (Recommended for Configuration)
**Status:** ✅ Analyzed, Recommended, Implemented
**Use Case:** Project configuration, schema definitions, pipeline settings

```toml
[metadata]
name = "GenomicVariant"
version = "1.0.0"
description = "Schema for genomic variant data with Indian population context"

[fields.chromosome]
type = "string"
required = true
enum = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "X", "Y", "MT"]

[fields.position]
type = "integer"
required = true
minimum = 1

[fields.population_frequencies.genome_india_af]
type = "number"
required = false
description = "GenomeIndia project allele frequency"
minimum = 0.0
maximum = 1.0

[fields.population_frequencies.indigen_af]
type = "number"
required = false
description = "IndiGen project allele frequency"
minimum = 0.0
maximum = 1.0
```

**Pros:**
- More human-readable than JSON
- Better for configuration files
- Supports comments and better organization
- Native Python support via `tomllib` (3.11+) or `toml` library

**Cons:**
- Not as widely supported as JSON
- Less tooling ecosystem than JSON Schema

### 2. JSON Schema (For API Validation)
**Status:** ✅ Analyzed, Recommended for APIs
**Use Case:** API request/response validation, OpenAPI documentation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "chromosome": {"type": "string", "enum": ["1","2",...,"X","Y","MT"]},
    "position": {"type": "integer", "minimum": 1},
    "ref_allele": {"type": "string", "pattern": "^[ACGTN]+$"}
  }
}
```

**Pros:**
- Industry standard for API validation
- Rich ecosystem of tools and libraries
- Excellent for OpenAPI/Swagger documentation

### 2. Protobuf (For High-Performance Data Transfer)
**Status:** ✅ Analyzed, Recommended for internal pipelines
**Use Case:** Internal data processing, ML pipelines, high-throughput analysis

```protobuf
syntax = "proto3";

message GenomicVariant {
  string chromosome = 1;
  uint64 position = 2;
  string ref_allele = 3;
  string alt_allele = 4;
  float quality = 5;

  message Annotation {
    string gene_symbol = 1;
    string protein_change = 3;
    Significance clinical_significance = 4;

    enum Significance {
      PATHOGENIC = 1;
      BENIGN = 2;
      UNCERTAIN = 3;
    }
  }

  repeated Annotation annotations = 7;

  message PopulationFrequency {
    float genome_india_af = 1;
    float indigen_af = 2;
  }

  PopulationFrequency population_frequencies = 8;
}
```

**Pros:**
- Compact binary format
- Fast serialization/deserialization
- Strongly typed
- Multi-language support

**Cons:**
- Schema evolution complexity
- Less human-readable

### 3. Avro (For Data Lake Integration)
**Status:** ✅ Analyzed, Recommended for storage
**Use Case:** Long-term storage, analytics, data lake queries

```avro
{
  "type": "record",
  "name": "GenomicVariant",
  "fields": [
    {"name": "chromosome", "type": "string"},
    {"name": "position", "type": "long"},
    {"name": "annotations", "type": {
      "type": "record",
      "name": "Annotation",
      "fields": [
        {"name": "gene_symbol", "type": ["string", "null"]},
        {"name": "protein_change", "type": ["string", "null"]}
      ]
    }}
  ]
}
```

**Pros:**
- Schema evolution support
- Hadoop/Spark ecosystem integration
- Good for Parquet storage

**Cons:**
- More verbose than Protobuf

## API Specification Approaches Analysis

### 1. OpenAPI 3.1 (RESTful APIs)
**Status:** ✅ Analyzed, Recommended for integrations
**Use Case:** Simple CRUD operations, ABDM integration, clinical workflows

```yaml
openapi: 3.1.0
info:
  title: Dirghayu Genomics API
  version: 1.0.0

paths:
  /variants/{variant_id}:
    get:
      summary: Get variant information
      parameters:
        - name: variant_id
          in: path
          required: true
          schema:
            type: string
            pattern: '^[0-9]+:[0-9]+:[ACGT]+:[ACGT]+$'
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Variant'

components:
  schemas:
    Variant:
      type: object
      properties:
        annotations:
          type: array
          items:
            $ref: '#/components/schemas/Annotation'
```

**Pros:**
- Industry standard
- Auto-generated documentation
- Tool ecosystem (Swagger UI, Postman)
- Good for clinical integrations

**Cons:**
- Verbose for complex genomic queries
- Multiple round trips for related data

### 2. GraphQL (For Complex Genomic Queries)
**Status:** ✅ Analyzed, Recommended for main API
**Use Case:** Clinical dashboards, research queries, complex relationships

```graphql
type Query {
  variant(id: ID!): Variant
  patient(id: ID!): Patient
  gene(symbol: String!): Gene
}

type Variant {
  id: ID!
  chromosome: String!
  annotations: [Annotation!]!
  populationFrequencies: PopulationFrequencies!
  pathogenicityScore: Float
}

type Patient {
  id: ID!
  variants(first: Int, after: String): VariantConnection!
  riskAssessments: [RiskAssessment!]!
}
```

**Pros:**
- Single endpoint for complex queries
- Client-specified data requirements
- Reduces over/under-fetching

**Cons:**
- Learning curve
- Schema versioning challenges

### 3. gRPC (For High-Performance ML Inference)
**Status:** ✅ Analyzed, Recommended for ML pipelines
**Use Case:** Real-time analysis, ML model serving, streaming data

```protobuf
service GenomicsService {
  rpc AnalyzeVariants(stream Variant) returns (stream VariantAnalysis);
  rpc PredictProteinStability(ProteinStructure) returns (StabilityPrediction);
  rpc CalculateRiskScores(RiskAssessmentRequest) returns (RiskAssessmentResponse);
}

message Variant {
  string chromosome = 1;
  uint64 position = 2;
  repeated Annotation annotations = 6;
}
```

**Pros:**
- High performance
- Streaming support
- Strong typing

**Cons:**
- HTTP/2 requirement
- Browser limitations

## Implementation Roadmap

### Phase 1: Core Data Models (Q1 2026)

#### Schema Definition
- [ ] Define JSON Schema for API validation
- [ ] Create Protobuf definitions for internal data structures
- [ ] Set up Avro schemas for data lake storage
- [ ] Implement schema registry (Confluent/Pulsar)

#### Core Entities
- [ ] Variant model with Indian population frequencies
- [ ] Patient/Individual model with clinical metadata
- [ ] Gene/Protein model with structural data
- [ ] Analysis/Report model for results
- [ ] Phenotype/Clinical data models

#### Validation Rules
- [ ] VCF format validation rules
- [ ] Clinical significance classification
- [ ] Quality control thresholds
- [ ] Indian-specific validation rules

### Phase 2: API Layer Development (Q2 2026)

#### GraphQL API
- [ ] Set up GraphQL server (Apollo Server/FastAPI-GraphQL)
- [ ] Define GraphQL schema for genomic queries
- [ ] Implement resolvers for variant queries
- [ ] Add authentication/authorization
- [ ] Implement query optimization and caching

#### REST API (OpenAPI)
- [ ] Create OpenAPI 3.1 specification
- [ ] Implement REST endpoints for CRUD operations
- [ ] Add FHIR resource compatibility
- [ ] Implement ABDM integration endpoints
- [ ] Add rate limiting and monitoring

#### gRPC Services
- [ ] Define gRPC service interfaces
- [ ] Implement ML model inference services
- [ ] Add streaming capabilities for large datasets
- [ ] Set up service mesh (Istio/Linkerd)

### Phase 3: Data Processing Pipeline (Q3 2026)

#### Ingestion Pipeline
- [ ] VCF file parsing and validation
- [ ] Annotation pipeline integration (VEP, custom annotators)
- [ ] Quality control and filtering
- [ ] Data partitioning and indexing

#### Storage Layer
- [ ] Set up Delta Lake for variant data
- [ ] Configure Parquet optimization
- [ ] Implement data partitioning strategies
- [ ] Add data compression and encryption

#### Analytics Pipeline
- [ ] Population frequency calculations
- [ ] PRS score computation
- [ ] Protein stability predictions
- [ ] Risk assessment algorithms

### Phase 4: Integration & Compliance (Q4 2026)

#### Indian Healthcare Integration
- [ ] ABDM API integration
- [ ] FHIR resource mapping
- [ ] Ayushman Bharat compatibility
- [ ] Regional language support

#### Regulatory Compliance
- [ ] ICMR data sharing guidelines
- [ ] HIPAA-like privacy protections
- [ ] Audit logging and traceability
- [ ] Consent management system

#### Security & Privacy
- [ ] End-to-end encryption
- [ ] Role-based access control
- [ ] Data anonymization pipelines
- [ ] Secure key management

## Technology Choices

### Data Storage
- **Primary:** Delta Lake on S3/Cloud Storage
- **Metadata:** PostgreSQL with TimescaleDB
- **Search:** Elasticsearch for variant queries
- **Cache:** Redis for frequently accessed data

### API Frameworks
- **GraphQL:** Apollo Server / Graphene-Python
- **REST:** FastAPI with Pydantic validation
- **gRPC:** gRPC Python with protocol buffers

### Infrastructure
- **Compute:** Kubernetes with GPU support
- **Workflows:** Nextflow/Snakemake for pipelines
- **Monitoring:** Prometheus + Grafana
- **Logging:** ELK stack

## Performance Considerations

### Query Optimization
- [ ] Implement database indexing strategies
- [ ] Add query result caching
- [ ] Optimize for common genomic queries
- [ ] Implement pagination for large result sets

### Data Compression
- [ ] Use Zstandard for variant data
- [ ] Implement columnar storage (Parquet)
- [ ] Add data deduplication
- [ ] Optimize for cloud storage costs

### Scalability
- [ ] Horizontal scaling for API services
- [ ] Auto-scaling for compute-intensive tasks
- [ ] CDN integration for static assets
- [ ] Multi-region deployment strategy

## Testing & Validation

### Schema Testing
- [ ] Unit tests for all schema validations
- [ ] Integration tests for data pipelines
- [ ] Performance benchmarks
- [ ] Compatibility testing across versions

### API Testing
- [ ] GraphQL query testing
- [ ] REST endpoint testing
- [ ] gRPC service testing
- [ ] Load testing and stress testing

## Documentation

### API Documentation
- [ ] Auto-generated OpenAPI documentation
- [ ] GraphQL schema documentation
- [ ] gRPC service documentation
- [ ] Integration guides for developers

### Data Dictionary
- [ ] Field definitions and constraints
- [ ] Data relationships and dependencies
- [ ] Quality control rules
- [ ] Update frequencies and versioning

## Risk Assessment

### Technical Risks
- **Schema Evolution:** Changes to data models could break existing analyses
- **Performance:** Genomic data volumes could overwhelm infrastructure
- **Integration Complexity:** Multiple healthcare systems with different standards

### Mitigation Strategies
- **Schema Versioning:** Semantic versioning with backward compatibility
- **Scalability Planning:** Cloud-native architecture from day one
- **Incremental Integration:** Start with pilot integrations, expand gradually

## Success Metrics

### Technical Metrics
- Query response times < 500ms for common operations
- 99.9% API uptime
- Support for 10,000+ concurrent users
- Data processing throughput: 1000 genomes/day

### Business Metrics
- ABDM integration completion
- Clinical partner adoption
- Research publication output
- Cost per genome analysis

## Next Steps

1. **Immediate (Week 1-2):**
   - Set up schema registry
   - Define core data models
   - Create API specification framework

2. **Short-term (Month 1-3):**
   - Implement GraphQL API
   - Build data ingestion pipeline
   - Set up basic storage layer

3. **Medium-term (Month 3-6):**
   - Add ML inference capabilities
   - Implement clinical integrations
   - Performance optimization

4. **Long-term (Month 6+):**
   - Scale to production levels
   - Add advanced analytics
   - Expand healthcare integrations

---

## Decision Summary

**Recommended Architecture:**
- **Data Layer:** Protobuf (internal) + Avro (storage) + JSON Schema (validation)
- **API Layer:** GraphQL (primary) + REST (integrations) + gRPC (ML)
- **Storage:** Delta Lake + PostgreSQL + Elasticsearch
- **Compute:** Kubernetes with GPU support

**Key Principles:**
- Start simple, scale complex
- Prioritize data integrity and privacy
- Design for Indian healthcare ecosystem
- Enable both research and clinical use cases

**Critical Success Factors:**
- Strong schema governance
- Performance-optimized queries
- Regulatory compliance
- Healthcare system integration