#!/bin/bash
source /root/miniconda3/bin/activate personal_database
cd /root/llm-universe/personal_knowledge_base
python -m pytest tests/ 