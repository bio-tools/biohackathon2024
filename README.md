# BioHackathon Europe 2024
ELIXIR Biohackathon 2024 project nr. 16

## Enhancing bio.tools by Semantic Literature Mining

## Setting up the project

### Prerequisites

- Python 3.11 or higher
- Poetry (can be installed via `pip install poetry`)

### Installation

1. Clone the repository
2. Run `poetry install` in the root directory of the repository
3. Run `poetry shell` to activate the virtual environment
4. Kernel installation: `python -m ipykernel install --user --name=biohackathon-2024`

## Project Description
This project aims to improve and extend bio.tools metadata through fine-tuned named-entity recognition (NER) from Europe PMC and other established literature mining software. This will help researchers find uses of particular software and measure the impact of research software beyond paper citations, thus providing a better indicator of their impact. Text mining mentions of software is a non-trivial problem, as the software often is homonymous with other entities, such as chemicals , genes or organisms. However, NER of software is facilitated by frequent context words such as “version”, “software” or “program”. 

This will be further exploited by integration of the often very detailed bio.tools annotations to enhance software recognition. We expect to identify ensembles of publications for thousands of software tools annotated in bio.tools, adding valuable information about tool usage to Europe PMC and providing relevant background data for more accurate and deeper tool categorization and annotations, as well as improved benchmarking of the tools themselves.

