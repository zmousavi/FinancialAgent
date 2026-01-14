#!/usr/bin/env python3
"""
05_create_embeddings.py

PURPOSE:
    Generate embeddings (vector representations) for document chunks using
    Google's Vertex AI text-embedding-004 model.

WHAT ARE EMBEDDINGS?
    Embeddings are numerical vectors that represent the meaning of text.
    Similar texts have similar vectors, which allows us to search by meaning.

    Example:
    "Apple's revenue" → [0.23, -0.45, 0.67, ...] (768 numbers)
    "Apple's income"  → [0.24, -0.43, 0.68, ...] (similar numbers!)

PIPELINE POSITION:
    04_create_chunks.py → [05_create_embeddings.py] → 06_setup_vector_db.py

WHY VERTEX AI?
    - Google's text-embedding-004 model (768 dimensions)
    - More cost-effective than OpenAI
    - Better integration with GCP infrastructure
    - Task types optimize for document search
"""

import os
import json
import yaml
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()


def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.yaml

    Returns:
        Dictionary with all config settings (embeddings, paths, etc.)
    """
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def initialize_vertex_ai(config: Dict[str, Any]) -> genai.Client:
    """
    Initialize connection to Google Vertex AI

    WHY THIS IS NEEDED:
        - Sets up authentication to GCP
        - Connects to the correct project and region
        - Creates client for making API calls

    Args:
        config: Configuration dictionary from config.yaml

    Returns:
        genai.Client configured for Vertex AI
    """
    # Get project ID from environment variable
    project_id = os.getenv(config['vertex_ai']['project_id_env_var'])
    location = config['vertex_ai']['location']

    if not project_id:
        raise ValueError(
            f"Environment variable {config['vertex_ai']['project_id_env_var']} "
            f"not set in .env file"
        )

    # Create client using the NEW google-genai SDK
    client = genai.Client(
        vertexai=True,  # Use Vertex AI (not public Gemini API)
        project=project_id,
        location=location
    )

    print(f"✓ Initialized Vertex AI")
    print(f"  Project: {project_id}")
    print(f"  Location: {location}")

    return client


def create_embeddings_batch(
    texts: List[str],
    client: genai.Client,
    model_name: str = "text-embedding-004"
) -> List[List[float]]:
    """
    Create embeddings for a batch of texts using Vertex AI

    WHAT THIS DOES:
        Takes a list of text strings and converts each to a 768-dimensional vector

    WHY BATCH?
        - More efficient than one-at-a-time
        - Reduces API calls (saves cost and time)
        - Can process up to 100 texts per call

    SMART SPLITTING:
        - If batch exceeds 20k token limit, automatically splits in half
        - Keeps splitting recursively until all pieces fit
        - Handles variable-length chunks automatically

    Args:
        texts: List of text strings to embed (max 100 per batch)
        client: Initialized Vertex AI client
        model_name: Embedding model to use (text-embedding-004)

    Returns:
        List of embedding vectors (each is a list of 768 floats)

    EXAMPLE:
        texts = ["Apple's revenue grew", "Microsoft profit increased"]
        embeddings = create_embeddings_batch(texts, client)
        # Returns: [[0.23, -0.45, ...], [0.18, -0.42, ...]]
    """
    try:
        # Call Vertex AI embedding API
        # task_type="RETRIEVAL_DOCUMENT" tells the model these are documents
        # to be searched (vs "RETRIEVAL_QUERY" for search queries)
        response = client.models.embed_content(
            model=model_name,
            contents=texts,
            config={
                'task_type': 'RETRIEVAL_DOCUMENT',  # Optimize for document storage
                'output_dimensionality': 768  # Standard dimension for text-embedding-004
            }
        )

        # Extract the embedding vectors from the response
        vectors = [embedding.values for embedding in response.embeddings]
        return vectors

    except Exception as e:
        error_msg = str(e)

        # Check if error is due to token limit being exceeded
        if 'token count' in error_msg.lower() or 'INVALID_ARGUMENT' in error_msg:

            # Special case: Single chunk is too long
            if len(texts) == 1:
                print(f"    ⚠️  Single chunk too long ({len(texts[0])} chars), truncating...")
                # Truncate to ~10k characters (roughly 7-8k tokens)
                truncated_text = texts[0][:10000]
                return create_embeddings_batch([truncated_text], client, model_name)

            # Multiple chunks: Split in half and process recursively
            mid = len(texts) // 2  # Find middle position
            print(f"    ⚠️  Batch too large ({len(texts)} chunks), splitting into {mid} + {len(texts)-mid}...")

            # Process first half (recursive call)
            vectors1 = create_embeddings_batch(texts[:mid], client, model_name)

            # Process second half (recursive call)
            vectors2 = create_embeddings_batch(texts[mid:], client, model_name)

            # Combine results (locally, no API call)
            return vectors1 + vectors2
        else:
            # Some other error, re-raise it
            print(f"✗ Error generating embeddings: {e}")
            raise


def load_chunks_from_file(chunk_file: str) -> List[Dict[str, Any]]:
    """
    Load chunks from JSON file created by 04_create_chunks.py

    CHUNK FILE FORMAT:
        [
            {
                "text": "Apple Inc. is a technology company...",
                "company": "AAPL",
                "section": "Business",
                "document_type": "10-K",
                ...
            },
            ...
        ]

    Args:
        chunk_file: Path to chunk JSON file

    Returns:
        List of chunk dictionaries
    """
    with open(chunk_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return chunks


def process_chunks_to_embeddings(
    chunks_dir: str,
    embeddings_output_dir: str,
    client: genai.Client,
    model_name: str,
    batch_size: int = 100,
    skip_existing: bool = True
) -> Dict[str, int]:
    """
    Process all chunk files and generate embeddings

    WHAT THIS DOES:
        1. Finds all *_chunks.json files in chunks/ directory
        2. For each file, reads the chunks
        3. Generates embeddings in batches
        4. Saves embeddings to embeddings/ directory
        5. Creates combined all_embeddings.json file

    WHY SKIP EXISTING?
        - If script fails midway, can resume without re-doing work
        - Saves API costs and time
        - Can add new companies without re-processing everything

    Args:
        chunks_dir: Directory containing chunk files (e.g., "chunks/")
        embeddings_output_dir: Where to save embeddings (e.g., "embeddings/")
        client: Vertex AI client
        model_name: Embedding model name
        batch_size: Number of chunks per API call (max 100)
        skip_existing: Skip files that already have embeddings

    Returns:
        Statistics dictionary with counts of files processed
    """
    # Create output directory if it doesn't exist
    os.makedirs(embeddings_output_dir, exist_ok=True)

    # Find all chunk files (both annual and quarterly)
    chunk_files = []
    for root, dirs, files in os.walk(chunks_dir):
        for file in files:
            if file.endswith('_chunks.json'):  # Only process JSON files
                chunk_files.append(os.path.join(root, file))

    chunk_files.sort()
    print(f"\n✓ Found {len(chunk_files)} chunk files to process\n")

    # Track statistics
    stats = {
        'files_processed': 0,
        'total_chunks': 0,
        'total_embeddings': 0,
        'files_skipped': 0
    }

    # Store all embeddings for combined file
    all_embedded_chunks = []

    # Process each chunk file
    for chunk_file in chunk_files:
        filename = Path(chunk_file).name
        company = filename.split('_')[0]  # Extract company ticker (e.g., "AAPL")

        # Determine output path (preserve annual/quarterly structure)
        if 'annual' in chunk_file:
            output_subdir = os.path.join(embeddings_output_dir, 'annual')
        elif 'quarterly' in chunk_file:
            output_subdir = os.path.join(embeddings_output_dir, 'quarterly')
        else:
            output_subdir = embeddings_output_dir

        os.makedirs(output_subdir, exist_ok=True)
        output_file = os.path.join(
            output_subdir,
            filename.replace('_chunks.json', '_embeddings.json')
        )

        # Skip if already exists (saves time and API costs)
        if skip_existing and os.path.exists(output_file):
            print(f"⏭️  Embeddings exist for {company}, skipping")
            stats['files_skipped'] += 1

            # Load existing embeddings for combined file
            with open(output_file, 'r') as f:
                existing = json.load(f)
                all_embedded_chunks.extend(existing)
                stats['total_embeddings'] += len(existing)
            continue

        print(f"📄 Processing {company} ({filename})...")

        # Load chunks from file
        chunks = load_chunks_from_file(chunk_file)
        stats['total_chunks'] += len(chunks)

        # Extract just the text for embedding
        texts = [chunk['text'] for chunk in chunks]

        # Process in batches with progress bar
        embedded_chunks = []

        # tqdm creates a progress bar showing "[====>     ] 45%"
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding {company}"):
            batch_texts = texts[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]

            # Generate embeddings with retry logic (in case of temporary API errors)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Call the embedding API
                    embeddings = create_embeddings_batch(
                        batch_texts,
                        client,
                        model_name
                    )

                    # Add embeddings to chunk metadata
                    for chunk, embedding in zip(batch_chunks, embeddings):
                        chunk_with_embedding = chunk.copy()
                        chunk_with_embedding['embedding'] = embedding
                        embedded_chunks.append(chunk_with_embedding)

                    break  # Success! Exit retry loop

                except Exception as e:
                    # If this isn't the last attempt, wait and retry
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        print(f"  ⚠️  Retry {attempt + 1}/{max_retries} after error: {e}")
                        print(f"  ⏳ Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        # Last attempt failed, give up
                        print(f"  ✗ Failed after {max_retries} attempts: {e}")
                        raise

        # Save individual file embeddings
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embedded_chunks, f, indent=2)

        print(f"  ✓ Saved {len(embedded_chunks)} embeddings to {output_file}\n")

        # Add to combined list
        all_embedded_chunks.extend(embedded_chunks)
        stats['files_processed'] += 1
        stats['total_embeddings'] += len(embedded_chunks)

    # Save combined embeddings file (required by 06_setup_vector_db.py)
    combined_output = os.path.join(embeddings_output_dir, 'all_embeddings.json')

    # Include metadata about the embeddings
    combined_data = {
        'chunks': all_embedded_chunks,
        'metadata': {
            'model': model_name,
            'provider': 'vertex-ai',
            'dimension': 768,
            'total_chunks': len(all_embedded_chunks),
            'task_type': 'RETRIEVAL_DOCUMENT'
        }
    }

    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)

    print(f"\n✓ Saved combined embeddings to {combined_output}")

    return stats


def main():
    """
    Main function to generate embeddings

    WORKFLOW:
        1. Load config from config.yaml
        2. Initialize Vertex AI connection
        3. Process all chunk files
        4. Generate embeddings in batches
        5. Save individual and combined embedding files
    """
    print("=" * 70)
    print("VERTEX AI EMBEDDING GENERATION")
    print("=" * 70)

    # Load configuration
    config = load_config()

    # Initialize Vertex AI
    client = initialize_vertex_ai(config)

    # Get settings from config
    model_name = "text-embedding-004"  # Google's latest embedding model
    chunks_dir = "chunks"  # Input directory
    embeddings_dir = config['output_dirs'].get('embeddings', 'embeddings')
    batch_size = config['embeddings'].get('batch_size', 100)

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Input: {chunks_dir}/")
    print(f"  Output: {embeddings_dir}/")
    print("-" * 70)

    # Process all chunks
    stats = process_chunks_to_embeddings(
        chunks_dir=chunks_dir,
        embeddings_output_dir=embeddings_dir,
        client=client,
        model_name=model_name,
        batch_size=batch_size,
        skip_existing=True  # Skip already-processed files
    )

    # Print final statistics
    print("\n" + "=" * 70)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 70)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files skipped: {stats['files_skipped']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total embeddings: {stats['total_embeddings']}")
    print(f"\n📊 Output: {embeddings_dir}/all_embeddings.json")
    print(f"\n➡️  Next step: Run 06_setup_vector_db.py to create searchable database")
    print("=" * 70)


if __name__ == "__main__":
    main()
