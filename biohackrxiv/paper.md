---
title: 'Enhancing bio.tools by Semantic Literature Mining'
tags:
  - text mining
  - bio.tools
  - Europe PMC
authors:
  - name: Aleksandra Szmigiel
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Ana Mendes
    orcid: 0009-0008-5170-0927
    affiliation: 1
  - name: Erik Jaaniso
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Magnus Palmblad
    orcid: 0000-0002-5865-8994
    affiliation: 3
  - name: Rob M. Ewing
    orcid: 0000-0001-6510-4001
    affiliation: 4  
  - name: Santosh Tirunagari
    orcid: 
    affiliation: 5
  - name: Tess Afanasyeva
    orcid: 0000-0003-3962-0779
    affiliation: 6
  - name: Vedran Kasalica
    orcid: 0000-0002-0097-1056
    affiliation: 7
  - name: Veit Schwämmle
    orcid: 0000-0002-9708-6722
    affiliation: 1
  - name: Zunaira Shafique
    orcid: 0000-0000-0000-0000
    affiliation: 5
affiliations:
 - name: University of Southern Denmark, Campusvej 55, 5230, Odense, Denmark
   index: 1
 - name:
   index: 2
 - name: Leiden University Medical Center, Albinusdreef 2, 2333 ZA, Leiden, The Netherlands
   index: 3
 - name: School of Biological Sciences, University of Southampton, Southampton, SO17 1BJ, UK
   index: 4
 - name:
   index: 5
 - name: Sanquin Research, Amsterdam, The Netherlands
   index: 6
 - name: Netherlands eScience Center, Science Park 402, 1098 XH, Amsterdam, The Netherlands
   index: 7
date: 05 November 2024
bibliography: paper.bib
authors_short: Last et al. Enhancing bio.tools by Semantic Literature Mining
group: BioHackrXiv
event: BioHackathon Europe 2024
---

# Background


## Software mentions in the scientific literature

The mining and study of mentions of software in the scientific literature has recenntly received much attention.

OpenAIRE...

The Chan Zuckerberg Initiative (CZI) has produced a software mentions dataset, CZ Software Mentions [@istrate2022largedatasetsoftwarementions], where the mentions were extracted with a trained SciBERT model from several sources: the NIH PubMed Central collection and from papers provided by various publishers to the Chan Zuckerberg Initiative.

Schindler et al. [@PMID:35111920] constructed a knowledge graph from software mentions in the scientific literature, differenting different types of software, such as applications, plugins, or operating systems, and types of mention, including usage, creation and deposition. The bio.tools corpus built primarily from primary publications would be expected to cover all types of mention, including describing the creation and availability (deposition) of the tool, as well as some use cases (usage).

Software synonyms - SciCrunch

Disambiguation (if anyone looked at this?)


## bio.tools

bio.tools [@PMID:31405382] is a comprehensive registry of bioinformatics tools and data services, designed to assist researchers in discovering, understanding, and utilizing software resources in the life sciences. Developed under the ELIXIR infrastructure, it serves as a centralized platform for accessing essential scientific and technical metadata about various bioinformatics tools, databases, and services. This metadata includes software licenses, operating system compatibility, repository URLs, but also semantic notation of the operation(s) performed by (or with) the software, and the data types and file format consumed and produced in the EDAM ontology [@PMID:23479348].


## bio.tools and the literature

We identify three distinct types of relationships between software tools (entries) in bio.tools and articles:

1. Articles associated with bio.tools entries in bio.tools, e.g. 10.1002/pmic.201200439 in https://bio.tools/comet
2. Articles citing articles associated with entries in bio.tools, e.g. 10.1002/rcm.9937 cites 10.1002/pmic.201200439 associated with https://bio.tools/comet
3. Articles mentioning tool in bio.tools, e.g. 10.1021/acs.jproteome.3c00902

For example, there are currently four distinct tools named "comet" with varying capitalization: Comet [@PMID:23148064], coMET [@PMID:25928765], CoMet [@PMID:21622656] and COMET [@PMID:32034124]. The first is a tool used in proteomics to match tandem mass spectra to peptides. The last three are used in genetics, metagenomics and epigenetics, respectivly. Since it cannot be assumed that capitalization is consistent, disambiguating homonyms such as these is a non-trivial task, but necessary for accurate mining of tool mentions and extracting information on the (co-)usage of software tools.


### Subsection level 3

Please keep sections to a maximum of three levels.

## Tables, figures and so on

Please remember to introduce tables (see Table 1) before they appear on the document. We recommend to center tables, formulas and figure but not the corresponding captions. Feel free to modify the table style as it better suits to your data.

Table 1
| Header 1 | Header 2 |
| -------- | -------- |
| item 1 | item 2 |
| item 3 | item 4 |

Remember to introduce figures (see Figure 1) before they appear on the document. 

![BioHackrXiv logo](./biohackrxiv.png)
 
Figure 1. A figure corresponding to the logo of our BioHackrXiv preprint.

# Other main section on your manuscript level 1

Feel free to use numbered lists or bullet points as you need.
* Item 1
* Item 2

# Discussion and/or Conclusion

We recommend to include some discussion or conclusion about your work. Feel free to modify the section title as it fits better to your manuscript.

# Future work

And maybe you want to add a sentence or two on how you plan to continue. Please keep reading to learn about citations and references.

For citations of references, we prefer the use of parenthesis, last name and year. If you use a citation manager, Elsevier – Harvard or American Psychological Association (APA) will work. If you are referencing web pages, software or so, please do so in the same way. Whenever possible, add authors and year. We have included a couple of citations along this document for you to get the idea. Please remember to always add DOI whenever available, if not possible, please provide alternative URLs. You will end up with an alphabetical order list by authors’ last name.

# Jupyter notebooks, GitHub repositories and data repositories

* Please add a list here
* Make sure you let us know which of these correspond to Jupyter notebooks. Although not supported yet, we plan to add features for them
* And remember, software and data need a license for them to be used by others, no license means no clear rules so nobody could legally use a non-licensed research object, whatever that object is

# Acknowledgements
Please always remember to acknowledge the BioHackathon, CodeFest, VoCamp, Sprint or similar where this work was (partially) developed.

# References

Leave thise section blank, create a paper.bib with all your references.
