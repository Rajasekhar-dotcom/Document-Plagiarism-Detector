# 📄 Scalable Document Plagiarism Engine

## Overview
Standard document comparison algorithms rely on 1-to-1 linear scanning, resulting in highly inefficient $O(N \cdot D)$ time complexities when searching against large datasets. This project is a highly scalable plagiarism detection engine built in C++ that solves the multi-document comparison bottleneck. 

By implementing the **Rabin-Karp rolling hash** alongside the **Winnowing algorithm**, the system generates highly compressed document "fingerprints." These fingerprints are stored in a global **Inverted Index** hash map. When a target document is submitted, the engine queries the inverted index, reducing search time to sub-second $O(W)$ lookups and calculating Jaccard similarity only for mathematically guaranteed candidate matches.

## Core Architecture

### 1. K-Gram Tokenization & Rabin-Karp Hashing
The engine cleans text and generates contiguous substrings of length $k$. To avoid the $O(k)$ cost of hashing each string from scratch, it uses a rolling hash:
$$H_{next} = ((H_{prev} - c_{old} \cdot b^{k-1}) \cdot b + c_{new}) \pmod{M}$$
This guarantees $O(1)$ hash computation per k-gram.

### 2. The Winnowing Algorithm

Instead of storing every hash, the engine groups hashes into windows of size $w$ and selects the minimum value. This mathematically guarantees that any matching substring of length $t = w + k - 1$ will be detected, while drastically reducing memory footprint.

### 3. The Inverted Index

Instead of scanning documents individually, the pre-computed fingerprints are stored in a hash map where `Key = Hash` and `Value = List of Document IDs`. This allows $O(1)$ lookups for overlapping k-grams.

## Installation & Compilation
```bash
# Clone the repository
git clone [https://github.com/yourusername/plagiarism-engine.git](https://github.com/yourusername/plagiarism-engine.git)
cd plagiarism-engine

# Compile using g++
g++ -O3 -std=c++17 src/main.cpp src/hasher.cpp src/winnowing.cpp -o engine

# Run the engine
./engine --corpus ./data/corpus_folder --target ./data/student_essay.txt
