Evaluation Agent POC

Base Model: qwen3:0.6b on ollama

Embedding Model: M3E

bert Model: iic/nlp_bert_document-segmentation_chinese-base



```
agent
├─ 📁.vscode
│  └─ 📄launch.json
├─ 📁bert
│  ├─ 📄.msc
│  ├─ 📄.mv
│  ├─ 📄added_tokens.json
│  ├─ 📄config.json
│  ├─ 📄configuration.json
│  ├─ 📄pytorch_model.bin
│  ├─ 📄special_tokens_map.json
│  ├─ 📄tokenizer.json
│  ├─ 📄tokenizer_config.json
│  ├─ 📄vocab.txt
│  ├─ 📄zh_exp_res.jpg
│  └─ 📄模型结构.jpg
├─ 📁config
│  ├─ 📄models.yaml
│  └─ 📄__init__.py
├─ 📁example_data
│  ├─ 📁interview
│  │  ├─ 📄1. Machine Learning Interview Questions
│  │  ├─ 📄2. Deep Learning Interview Questions
│  │  ├─ 📄5. Behavioral_HR Interview Questions.pdf
│  │  ├─ 📄A to Z Python Cheatsheet.html
│  │  ├─ 📄Data_Science_Resume_Checklist_Data_Interview_Pro.pdf
│  │  ├─ 📄deeplearning_questions.csv
│  │  ├─ 📄LICENSE
│  │  └─ 📄README.md
│  ├─ 📁jobs
│  │  ├─ 📄data.json
│  │  ├─ 📄jobs_dataset.csv
│  │  └─ 📄jobs_dataset.json
│  ├─ 📄H3C.pdf
│  ├─ 📄qa_test.json
│  ├─ 📄repair.pdf
│  ├─ 📄sample_qa_dataset.csv
│  └─ 📄updated_qa_dataset.csv
├─ 📁index
│  ├─ 📄data_description.index
│  ├─ 📄data_positionName.index
│  ├─ 📄H3C.index
│  └─ 📄repair.index
├─ 📁m3e-small
│  ├─ 📁1_Pooling
│  │  └─ 📄config.json
│  ├─ 📄.gitattributes
│  ├─ 📄config.json
│  ├─ 📄modules.json
│  ├─ 📄pytorch_model.bin
│  ├─ 📄README.md
│  ├─ 📄sentence_bert_config.json
│  ├─ 📄special_tokens_map.json
│  ├─ 📄tokenizer.json
│  └─ 📄tokenizer_config.json
├─ 📁models
│  ├─ 📁__pycache__
│  │  ├─ 📄auzre.cpython-310.pyc
│  │  ├─ 📄ollama.cpython-310.pyc
│  │  └─ 📄__init__.cpython-310.pyc
│  ├─ 📄auzre.py
│  ├─ 📄ollama.py
│  └─ 📄__init__.py
├─ 📁templates
│  └─ 📄ai_studio_code.html
├─ 📁tools
│  ├─ 📁__pycache__
│  │  ├─ 📄jobs_match.cpython-310.pyc
│  │  ├─ 📄match.cpython-310.pyc
│  │  ├─ 📄pdf_retrieval.cpython-310.pyc
│  │  ├─ 📄retrieval.cpython-310.pyc
│  │  └─ 📄__init__.cpython-310.pyc
│  ├─ 📄jobs_match.py
│  ├─ 📄pdf_retrieval.py
│  └─ 📄__init__.py
├─ 📁utils
│  ├─ 📁__pycache__
│  │  ├─ 📄build_client.cpython-310.pyc
│  │  ├─ 📄convert.cpython-310.pyc
│  │  ├─ 📄json_process.cpython-310.pyc
│  │  ├─ 📄pdf_preocess.cpython-310.pyc
│  │  ├─ 📄pdf_process.cpython-310.pyc
│  │  ├─ 📄qa_rag.cpython-310.pyc
│  │  ├─ 📄rag.cpython-310.pyc
│  │  ├─ 📄toolscalls_process.cpython-310.pyc
│  │  ├─ 📄vector_search.cpython-310.pyc
│  │  ├─ 📄vector_store.cpython-310.pyc
│  │  └─ 📄__init__.cpython-310.pyc
│  ├─ 📄build_client.py
│  ├─ 📄csv_utils.py
│  ├─ 📄json_process.py
│  ├─ 📄pdf_process.py
│  ├─ 📄qa_rag.py
│  ├─ 📄toolscalls_process.py
│  ├─ 📄vector_search.py
│  ├─ 📄vector_store.py
│  └─ 📄__init__.py
├─ 📄.gitignore
├─ 📄condaenv.sfno6v5n.requirements.txt
├─ 📄env.yml
├─ 📄main.py
├─ 📄MIGRATION_GUIDE.md
├─ 📄readme.md
├─ 📄test.ipynb
└─ 📄test_qa_rag.py
```