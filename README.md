# Enhancing bio.tools by Semantic Literature Mining (BioHackathon Europe 2024)

ELIXIR Biohackathon 2024 project nr. 16

## Setting up the Project

### Prerequisites

- **Python 3.11 or higher**
- **Poetry** (can be installed via `pip install poetry`)

### Setup Instructions

To get started with this project, please follow these steps to set up the environment and configure Jupyter for notebook use:

1. **Clone the Repository**  
   Clone the project repository to your local machine.

2. **Install Dependencies**  
   In the root directory of the repository, run the following command to install dependencies:

   `poetry install`

3. **Activate the Virtual Environment**  
   Activate the virtual environment created by Poetry with:

   `poetry shell`

4. **Configure Jupyter Notebook Kernel**  
   To use this environment in Jupyter notebooks, install a custom kernel by running:

   `python -m ipykernel install --user --name=biohackathon-2024 --display-name "BioHackathon 2024"`

   This command makes the environment available in Jupyter Notebook under the kernel name **BioHackathon 2024**

## Project Description

This project aims to improve and extend bio.tools metadata through fine-tuned named-entity recognition (NER) from Europe PMC and other established literature mining software. This will help researchers find uses of particular software and measure the impact of research software beyond paper citations, thus providing a better indicator of their impact. Text mining mentions of software is a non-trivial problem, as the software often is homonymous with other entities, such as chemicals , genes or organisms. However, NER of software is facilitated by frequent context words such as “version”, “software” or “program”.

This will be further exploited by integration of the often very detailed bio.tools annotations to enhance software recognition. We expect to identify ensembles of publications for thousands of software tools annotated in bio.tools, adding valuable information about tool usage to Europe PMC and providing relevant background data for more accurate and deeper tool categorization and annotations, as well as improved benchmarking of the tools themselves.
