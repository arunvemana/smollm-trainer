# SmolLM Note Summarizer Fine-Tuning

This repository contains the environment, datasets, and scripts used to fine-tune the SmolLM-135M
model specifically for summarizing personal notes and web links.

## 🎯 Objective

To take raw, messy notes (including URLs and task lists) and transform them into structured, readable
summaries.

Example:
* Input: Notes: [Train slm documents, https://youtube.com/watch?v=guide456]
* Output: A task to train slm documents. The note references a video tutorial or resource.

## 📂 Project Structure

1 .
2 ├── workspace/               # Raw and processed JSONL training data
3 ├── models/             # Local storage for base weights and LoRA adapters
4 ├── scripts/            # Parse, validate dataset and Training, inference scripts
5 └── README.md           # You are here!

