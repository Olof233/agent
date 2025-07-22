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
├─ 📁index
│  └─ 📄H3C.index
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
├─ 📁tools
│  ├─ 📁__pycache__
│  │  ├─ 📄retrieval.cpython-310.pyc
│  │  └─ 📄__init__.cpython-310.pyc
│  ├─ 📄retrieval.py
│  └─ 📄__init__.py
├─ 📁utils
│  ├─ 📁__pycache__
│  │  ├─ 📄build_client.cpython-310.pyc
│  │  ├─ 📄convert.cpython-310.pyc
│  │  ├─ 📄toolscalls_process.cpython-310.pyc
│  │  └─ 📄__init__.cpython-310.pyc
│  ├─ 📄build_client.py
│  ├─ 📄convert.py
│  ├─ 📄toolscalls_process.py
│  └─ 📄__init__.py
├─ 📄.gitignore
├─ 📄env.yml
├─ 📄H3C.pdf
├─ 📄main.py
├─ 📄readme.md
└─ 📄test.ipynb
```
