# tei-chunker
Document chunker specialized for TEI XML (i.e. GROBID outputs from academic PDF parsing)

https://www.tei-c.org/  
https://github.com/kermitt2/grobid

```
tei-chunker/
├── .github/
│   └── workflows/
│       ├── docker.yml
│       └── publish.yml
├── examples/
│   └── github-workflow.yml
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_chunking.py
│   ├── test_document.py
│   └── test_github_utils.py
├── tei_chunker/
│   ├── __init__.py
│   ├── __about__.py
│   ├── chunking.py
│   ├── document.py
│   ├── github_utils.py
│   └── service.py
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
└── pyproject.toml
```

