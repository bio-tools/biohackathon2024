import csv
import os
import re
import time
import unicodedata
from enum import Enum
from urllib.parse import urlparse, parse_qsl
from html import unescape

import requests
from bs4 import BeautifulSoup
from lxml import etree

from bh24_literature_mining.utils import load_biotools_from_zip


MAX_WORDS = 8
MAX_WORDS_COMPOUND = 4

EXCLUDED_ACRONYMS_IN_TOOL_NAME = {'EBI', 'NIH', 'SIB', 'ExPASy'}

EXCLUDED_NOT_SIMPLE = {
    # no cites
    'ChIA-PET', 'LiftOver', 'BioC', 'CRISPRCas', 'COVID-19 Dashboard',
    # has cites
    't-SNE', 'ChIP-Seq', 'circRNA', 'XGBoost', 'scRNA-seq', 'ceRNA', 'ROC Curve', 'cfDNA', 'IsomiR', 'PCHi-C',
    'SARS-CoV-2', 'MinION', 'sRNA', 'nCoV', 'HotSpot', 'ModeRNA', 'GeneView', 'scATAC-seq', 'aCGH', 'microRNA'
}
EXCLUDED_NOT_SIMPLE_PLURAL = {e + 's' for e in EXCLUDED_NOT_SIMPLE}
INCLUDED_SIMPLE = {
    # no cites
    'ggplot2', 'scikit-learn', 'fastqc', 'Matplotlib', 'numpy', 'matplotlib', 'Cutadapt', 'Human Phenotype Ontology',
    'cutadapt', 'Trim Galore', 'pheatmap', 'American Type Culture Collection', 'tabix', 'Open Targets', 'wgsim',
    'scikit-image', 'bbmap', 'Cellosaurus', 'ggtree', 'rstudio', 'Konstanz Information Miner', 'Google Colab', 'ICD-9',
    'Pipeline Pilot', 'LASTZ', 'Apache Hadoop', 'CONCOCT', 'cellosaurus', 'Antimicrobial Peptide Database', 'Squidpy',
    'Exome Variant Server', 'ea-utils', 'barrnap', 'European Variation Archive', 'sourmash', 'gffread', 'bbtools',
    'BBKNN', 'Golm Metabolome Database', 'Filtlong', 'pysster', 'SPIEC-EASI', 'Barrnap', 'Merqury', 'MAKER-P',
    'ICELLNET', 'Partek Genomics Suite', 'Galaxy Tools', 'repeatmasker', 'strobemers', 'rjags', 'Glimma', 'Nextclade',
    'getorf', 'cellchat', 'diffcyt', 'cellxgene', 'bwtool', 'pyfaidx', 'gffcompare', 'Shiny-Seq', 'immunarch',
    'motifmatchr', 'Parsnp', 'shovill', 'helixturnhelix',
    # no cites lower
    'cikitlearn', 'ggplot', 'numpy', 'fastqc', 'matplotlib', 'rstudio', 'rdkit', 'repeatmasker', 'cutadapt', 'biocarta',
    'tabix', 'trimgalore', 'bbmap', 'pheatmap', 'wgsim', 'dgidb', 'scikitimage', 'chemspider', 'cellosaurus', 'ggtree',
    'fastani', 'cellchat', 'bbtools', 'american type culture collection atcc', 'imagemagick', 'helixturnhelix',
    'gffread', 'google colab', 'somaticsniper', 'metagenemark', 'gffcompare', 'casoffinder', 'gemsim', 'genomestudio',
    'singlecellsignalr', 'open phacts', 'squidpy', 'varibench', 'abricate', 'picardtools', 'eautils', 'barrnap',
    'consurfdb', 'mnepython', 'bbknn', 'uniprot id mapping', 'optitype', 'cellxgene', 'sourmash', 'spieceasi',
    'nextstrainorg', 'plasflow', 'swisslipids', 'pysster', 'golm metabolome database', 'hicanu', 'jsphylosvg',
    'mirmine', 'europepmc', 'spatiallibd', 'neuromorphoorg', 'fastq screen', 'sympy', 'glimma', 'windowmasker',
    'covglue', 'amrfinderplus', 'netnglyc', 'protscale', 'dsysmap', 'kofamscan', 'doubletdetection',
    # has cites
    'Gene Ontology', 'Reactome', 'MATLAB', 'Entrez Gene', 'European Nucleotide Archive', 'Trimmomatic', 'Rfam',
    'PROSITE', 'Bioconda', 'Nucleotide Sequence Database Collaboration', 'PHYLIP', 'Mouse Genome Database', 'Orphanet',
    'Mouse Genome Informatics', 'bioconda', 'CORUM', 'ICD-10', 'seqtk', 'Ontology Lookup Service',
    'Rat Genome Database', 'MODOMICS', 'Newbler', 'ATTED-II', 'ICD-11',
    # has cites lower
    'bioconductor', 'genbank', 'pubchem', 'pymol', 'rcsb protein data bank', 'chembl', 'uniprotkb swissprot',
    'trimmomatic', 'flybase', 'snakemake', 'pubmed central', 'dbgap', 'bioconda', 'wikipathways',
    'the biogrid interaction database', 'biocyc', 'bioperl', 'wormbase', 'metacyc', 'orphanet', 'genenamesorg',
    'cytoscapejs', 'insdc', 'ngl viewer', 'lncrnadisease', 'tigrfams', 'fastxtoolkit', 'mouse genome informatics mgi',
    'the immune epitope database iedb', 'ecocyc', 'disprot', 'orthodb', 'regulondb', 'openms', 'mzmine', 'metabolights',
    'biojava', 'pubtator', 'dbptm', 'consurf',
    # 4 letters upper/lower
    'HGNC', 'ATCC', 'BLAT', 'DDBJ', 'kegg', 'IEDB', 'HMDD', 'ZFIN', 'EMDB', 'SBGN', 'gatk', 'pfam', 'BMRB', 'GBIF',
    'MNDR', 'GTRD', 'omim', 'pdbe'
}
# TODO add more
EXCLUDED_SIMPLE = {
    # lower
    'timecourse', 'biomodels', 'allpaths', 'dirichletmultinomial', 'tscan', 'beadarray', 'vectorbase',
    'variantannotation', 'scrna', 'one to three',
    # 4 letters upper
    'GSEA', 'SBML', 'DSSP', 'HPRD', 'MLST', 'GSVA', 'GMAP', 'SPIA', 'HSSP', 'RMAP', 'PGAP', 'MSEA', 'PLEK', 'RPBS',
    'MISA', 'IMPC', 'CRAC'
}

EXCLUDED_MATCHES = {'RNA', 'rna', 'DNA', 'dna'}

EXCLUDED_SOFTWARE_CATALOGUE_LINKS = {
    'http://ms-utils.org', 'http://cbs.dtu.dk/services', 'http://www.cbs.dtu.dk/services', 'http://expasy.org',
    'https://www.expasy.org/', 'http://bioconductor.org/'
}
SKIP_PATHS = {
    'packages', 'release', 'devel', 'bioc', 'src', 'contrib', 'archive', 'html', 'manuals', 'vignettes', 'pub',
    'resources', 'software', 'downloads', 'rel', 'apps', 'menu', 'embdoc', 'cgi', 'r', 'research', 'cgi-bin', 'tools',
    'bitbucket', 'confluence', 'index.php', 'web', 'tools', 'app', 'container', 'articles',
    'tool_runner?tool_id=toolshed.g2.bx.psu.edu', 'tool_runner?tool_id=toolshed.pasteur.fr', 'repos', 'wiki', 'forum',
    'repository', 'docker', 'listinfo', 'gitlab', 'mediawiki', 'github', 'view', 'root?tool_id=toolshed.g2.bx.psu.edu',
    'content', 'matlabcentral', 'articles', 'bioinfo', 'main', 'databases', 'services', 'resources', 'programs', 'en',
    'fr', 'group', 'people', 'sites', 'default', 'd', 'main'
}

def normalize_pub_title(title):
    return ''.join(c for c in title.lower() if c.isalnum())

EXCLUDED_CITES = {
    '25633503', 'pmc4509590', '10.1038/nmeth.3252', normalize_pub_title('Orchestrating high-throughput genomic analysis with Bioconductor'), # Bioconductor
    '27137889', 'pmc4987906', '10.1093/nar/gkw343', normalize_pub_title('The Galaxy platform for accessible, reproducible and collaborative biomedical analyses: 2016 update'), # Galaxy
    '10.7490/f1000research.1114334.1', normalize_pub_title('A public Galaxy platform at Pasteur used as an execution engine for web services'), # Galaxy Pasteur
    '10827456', '10.1016/s0168-9525(00)02024-2', normalize_pub_title('EMBOSS: The European Molecular Biology Open Software Suite'), # EMBOSS
    '10.1017/cbo9781139151405', normalize_pub_title('EMBOSS Developer\'s Guide'), # EMBOSS
    '10.1017/cbo9781139151399', normalize_pub_title('EMBOSS Administrator\'s Guide'), # EMBOSS
    '35412617', 'pmc9252731', '10.1093/nar/gkac240', normalize_pub_title('Search and sequence analysis tools services from EMBL-EBI in 2022'), # EMBL-EBI
    '26673705', 'pmc4702932', '10.1093/nar/gkv1352', normalize_pub_title('The European Bioinformatics Institute in 2016: Data growth and integration'), # EMBL-EBI
    '26687719', 'pmc4702834', '10.1093/nar/gkv1157', normalize_pub_title('Ensembl 2016'), # Ensembl
    '10592169', 'pmc102437', '10.1093/nar/28.1.10', normalize_pub_title('Database resources of the National Center for Biotechnology Information'), # NCBI
    '16381840', 'pmc1347520', '10.1093/nar/gkj158', normalize_pub_title('Database resources of the National Center for Biotechnology Information'), # NCBI
    '18940862', 'pmc2686545', '10.1093/nar/gkn741', normalize_pub_title('Database resources of the National Center for Biotechnology Information'), # NCBI
    '21097890', 'pmc3013733', '10.1093/nar/gkq1172', normalize_pub_title('Database resources of the National Center for Biotechnology Information'), # NCBI
    '24259429', 'pmc3965057', '10.1093/nar/gkt1146', normalize_pub_title('Database resources of the National Center for Biotechnology Information') # NCBI
}
EXCLUDED_CITES_EXCEPTION_IDS = {'bioconductor', 'galaxy', 'emboss', 'ebi_tools', 'ensembl', 'ncbi_resources'}


class Tokens(Enum):
    # ORIG = 'orig'
    CASED = 'cased'
    LOWER = 'lower'


def get_tokens(text):
    text = text.replace('/', ' ') # TODO en-dash, em-dash
    return text.split()


def get_tokens_cased(tokens):
    pattern = re.compile(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9+.,\-_:;()]+|[^a-zA-Z0-9+]+$') # punctuation from biotoolsSchema
    return [pattern.sub('', token) for token in tokens]


def lower(token):
    return ''.join(char for char in unicodedata.normalize('NFD', token) if char.isalnum()).lower().rstrip('0123456789')


def get_tokens_lower(tokens):
    return [lower(token) for token in tokens]


def get_conc_list(tokens, n):
    conc_list: list[(str, list[int], bool)] = []
    for i in range(0, len(tokens) - n):
        to_join = [tokens[j] for j in range(i, i + n + 1) if tokens[j]]
        if len(to_join) == n + 1:
            token = ' '.join(to_join)
            conc_list.append((token, range(i, i + n + 1), False))
            if n < MAX_WORDS_COMPOUND:
                for j in range(i + 1, i + n + 1):
                    token_compound = ' '.join(tokens[k] for k in range(i, j) if tokens[k]) + ' '.join(tokens[k] for k in range(j, i + n + 1) if tokens[k])
                    conc_list.append((token_compound, range(i, i + n + 1), True))
    return conc_list


def get_matches_tool(name, id, tokens_words, tokens_set, matches, find):
    if len([c for c in name if c.isalpha()]) == 0:
        return
    tokens_orig = name.split()
    tool = {Tokens.CASED: get_tokens_cased(tokens_orig), Tokens.LOWER: get_tokens_lower(tokens_orig)}
    tool_str = {}
    for token_label in Tokens:
        tool_str[token_label] = {' '.join([t for t in tool[token_label] if t])}
    n = len(tool[Tokens.CASED]) - 1
    if n < MAX_WORDS_COMPOUND:
        for token_label in Tokens:
            for i in range(1, n + 1):
                tool_str[token_label].add(' '.join(tool[token_label][j] for j in range(0, i) if tool[token_label][j]) + ' '.join(tool[token_label][j] for j in range(i, n + 1) if tool[token_label][j]))
    hyphenation_pattern = re.compile(r'-[a-z0-9.-]+$')
    for token_label in Tokens:
        for i in range(0, MAX_WORDS):
            for s in tool_str[token_label] & tokens_set[token_label][i]:
                for w in tokens_words[token_label][i]:
                    if s == w[0]:
                        matches[token_label].add((s, id, w[1], find if find else w[2]))
                    elif '-' in w[0]:
                        wn = hyphenation_pattern.sub('', w[0])
                        if s == wn:
                            matches[token_label].add((s, id, w[1], find if find else w[2]))


def get_matches(tokens, biotools):
    tokens_words = {}
    tokens_set = {}
    for token_label in Tokens:
        tokens_words[token_label] = []
        tokens_set[token_label] = []
        for i in range(0, MAX_WORDS):
            conc_list = get_conc_list(tokens[token_label], i)
            tokens_words[token_label].append(conc_list)
            tokens_set[token_label].append({t[0] for t in conc_list})

    matches: dict[Tokens, set[(str, str, list[int], bool)]] = {}
    for token_label in Tokens:
        matches[token_label] = set()
    for tool in biotools.values():
        matches_tool = set()
        name = tool['name'].strip()
        matches_tool.add((name, False))
        name_without_version = re.sub(r' [v0-9.]+$', '', name)
        matches_tool.add((name_without_version, False))
        name_split = re.split(r': | - ', name_without_version)
        if len(name_split) > 1:
            matches_tool.add((name_split[0], False))
        name_split = name_split[-1]
        match = re.search(r'\(([^)]+)\)', name_split)
        if match:
            name_inside = match.group(1)
            if len([c for c in name_inside if c.isupper()]) > 0 and name_inside not in EXCLUDED_ACRONYMS_IN_TOOL_NAME:
                matches_tool.add((name_inside, False))
        name_outside = re.sub(r'\s*\([^)]*\)\s*', ' ', name_split).strip()
        matches_tool.add((name_outside, False))
        name_acronym = ''.join(c for c in name_outside if c.isupper())
        if name_acronym != name_outside and len(name_acronym) > 2:
            matches_tool.add((name_acronym, True))
        for part in name_outside.split():
            if part != name_outside:
                part_no_lower = ''.join(c for c in part if not c.islower())
                if len(part_no_lower) > 1 and len([c for c in part_no_lower if c.isupper()]) > 0:
                    matches_tool.add((part, True))
        id = tool['biotoolsID']
        matches_tool.add((id, False))
        matches_tool.add((id.replace('_', ' '), False))
        for name, find in matches_tool:
            get_matches_tool(name, id, tokens_words, tokens_set, matches, find)
        # TODO synonyms (from SciCrunch API)
        # TODO approximate matching (insertion, deletion, substitution, transposition), use rapidfuzz.process.cdist

    return matches


def normalize_doi(doi):
    doi = doi.strip().lower()
    for prefix in 'doi:', 'http://doi.org/', 'https://doi.org/', 'http://dx.doi.org/', 'https://dx.doi.org/':
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi


def get_ids_pub(soup):
    ids_pub = {}
    for id_label in ['pmid', 'pmcid', 'doi']:
        ids_pub[id_label] = set()
        ids_pub[id_label].update([tag.text.strip().lower() for tag in soup.find_all('pub-id', {'pub-id-type': id_label})])
        ids_pub[id_label].update([tag.text.strip().lower() for tag in soup.find_all('article-id', {'pub-id-type': id_label})])
        if id_label == 'doi':
            ids_pub[id_label] = {normalize_doi(id_pub) for id_pub in ids_pub[id_label]}
    ids_pub['title'] = {normalize_pub_title(tag.text) for tag in soup.find_all('article-title')}
    return ids_pub


def find_ref(match, pub_titles, biotools, ids_pub):
    found = False
    tool = biotools[match[1]]
    ids = {}
    id_labels = ['pmid', 'pmcid', 'doi']
    for id_label in id_labels:
        ids[id_label] = set()
    ids['title'] = set()
    for publication in tool['publication']:
        for id_label in id_labels:
            if publication[id_label]:
                if id_label == 'doi':
                    id = normalize_doi(publication[id_label])
                else:
                    id = publication[id_label].strip().lower()
                ids[id_label].add(id)
                if id in pub_titles:
                    ids['title'].add(pub_titles[id][1])
    for id_label, v in ids.items():
        for id in v:
            if id in EXCLUDED_CITES and match[1] not in EXCLUDED_CITES_EXCEPTION_IDS:
                continue
            if id in ids_pub[id_label]:
                found = True
                break
        if found:
            break
    return found


def get_ref_ids(pub_titles, biotools, ids_pub):
    ids = {}
    for tool in biotools.values():
        found = False
        for publication in tool['publication']:
            for id_label in ['pmid', 'pmcid', 'doi']:
                if publication[id_label]:
                    if id_label == 'doi':
                        id = normalize_doi(publication[id_label])
                    else:
                        id = publication[id_label].strip().lower()
                    if id in ids_pub[id_label] or (id in pub_titles and pub_titles[id][1] in ids_pub['title']):
                        ids[tool['biotoolsID']] = id
                        found = True
                        break
            if found:
                break
    return ids


def get_find_url_tokens(tokens):
    find_url_tokens = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower.startswith('www.'):
            token_lower = token_lower[4:]
        if token_lower.startswith('www'):
            token_lower = token_lower[3:]
        find_url_tokens.append(token_lower)
    return find_url_tokens


def normalize_path(path):
    path = path.split('#')[0]
    return path


def normalize_path_aggressive(path):
    path = path.split('#')[0]
    path = path.split('?')[0]
    path = path.removesuffix('html')
    path = path.removesuffix('htm')
    path = path.removesuffix('.')
    return path


def find_url(tokens, match, biotools, find_url_tokens):
    found = 0
    tool = biotools[match[1]]
    urls_parsed = set()
    urls_parsed.add(urlparse(tool['homepage']))
    for link_label in ['link', 'documentation', 'download']:
        for link in tool[link_label]:
            if 'Software catalogue' not in link['type'] or link['url'] not in EXCLUDED_SOFTWARE_CATALOGUE_LINKS:
                urls_parsed.add(urlparse(link['url']))
    for url_parsed in urls_parsed:
        domain = ''.join(char for char in url_parsed.netloc if char.isalnum()).lower()
        paths = [path.lower() for path in url_parsed.path.split('/')[1:]]
        if paths and url_parsed.query:
            paths[-1] += '?' + url_parsed.query
        if domain.startswith('www.'):
            domain = domain[4:]
        if domain.startswith('www'):
            domain = domain[3:]
        for i in range(len(find_url_tokens)):
            token_lower = find_url_tokens[i]
            if domain == token_lower:
                found_1 = False
                found_2 = True
                paths_seen = 0
                match_lower = match[0].lower()
                for j in range(len(paths)):
                    if i + 1 + j == len(tokens):
                        break
                    if paths[j] not in SKIP_PATHS:
                        paths_seen += 1
                        token_lower = tokens[i + 1 + j].lower()
                        if match_lower == token_lower or match_lower == normalize_path_aggressive(token_lower) or match_lower in {v[1] for v in parse_qsl(token_lower)}:
                            found_1 = True
                        if paths[j] != token_lower and normalize_path(paths[j]) != normalize_path(token_lower):
                            found_2 = False
                        if paths_seen == 2:
                            break
                if found_2:
                    found = 2
                elif found_1:
                    found = 1
                if found > 1:
                    break
        if found > 1:
            break
    return found


def find_match(tokens, match, pub_titles, biotools, ids_pub, find_url_tokens):
    if match[0] in EXCLUDED_MATCHES:
        return 0
    if find_ref(match, pub_titles, biotools, ids_pub):
        return 3
    return find_url(tokens, match, biotools, find_url_tokens)


def is_simple(name):
    lower_seen = False
    upper_seen = False
    capital_seen = False
    first = True
    for name in name.split():
        for name in name.split('-'):
            if name[:-1].isupper() and name[-1] == 's':
                name = name[:-1]
            index = len(name) - 1
            while index >= 0 and name[index].isdigit():
                index -= 1
            name = name[:index + 1]
            if not name:
                continue
            if name.isalpha():
                if not name.islower() and not name.isupper() and not (name[0].isupper() and name[1:].islower()):
                    return False
            else:
                return False
            if first:
                first = False
                if name.islower():
                    lower_seen = True
            if name.isupper():
                upper_seen = True
            if name[0].isupper() and name[1:].islower():
                capital_seen = True
    if lower_seen and upper_seen or lower_seen and capital_seen or upper_seen and capital_seen:
        return False
    return True


def filter_matches(tokens, matches, cites, pub_titles, counts, wordlist, biotools, ids_pub, find_url_tokens):
    matches_filtered: set[(str, str, list[int], bool)] = set()
    matches_pass: dict[(str, str, bool), int] = {}
    matches_fail: set[(str, str, bool)] = set()
    for match in matches:
        if (match[0], match[1], match[3]) in matches_pass:
            matches_filtered.add(match + (matches_pass[(match[0], match[1], match[3])],))
            continue
        if (match[0], match[1], match[3]) in matches_fail:
            continue
        passes = False
        if not match[3]:
            if is_simple(match[0]):
                if match[0] in INCLUDED_SIMPLE:
                    passes = True
                elif (not match[0] in EXCLUDED_SIMPLE and (not match[0].islower() and len(match[0]) > 3 or len(match[0]) > 4)
                      and (match[0] not in counts or counts[match[0]] <= 1 or (match[1] in cites and cites[match[1]] > 0 and counts[match[0]] / cites[match[1]] / counts[''] < 0.000012))):
                    if match[0].isupper() and len(match[0]) == 4:
                        if match[0].isalpha() and not match[0].lower() in wordlist:
                            passes = True
                    elif not match[0][1:].islower() or any(c.isdigit() for c in match[0]):
                        passes = True
                    else:
                        word_count = 0
                        for word in match[0].split():
                            for subword in word.split('-'):
                                word_count += 1
                                if word_count > 2 or subword.lower() not in wordlist:
                                    passes = True
            else:
                if not match[0] in EXCLUDED_NOT_SIMPLE and not match[0] in EXCLUDED_NOT_SIMPLE_PLURAL and len(match[0]) > 3:
                    passes = True
        if passes:
            matches_filtered.add(match + (0,))
            matches_pass[(match[0], match[1], match[3])] = 0
        else:
            find_match_score = find_match(tokens, match, pub_titles, biotools, ids_pub, find_url_tokens)
            if find_match_score:
                matches_filtered.add(match + (find_match_score,))
                matches_pass[(match[0], match[1], match[3])] = find_match_score
            else:
                matches_fail.add((match[0], match[1], match[3]))
    return matches_filtered


def fill_result(match, results, filled, matches_unused, tokens, pub_titles, biotools, ids_pub, find_url_tokens):
    if match in matches_unused:
        del matches_unused[match]
    overlap = {}
    for fill in filled:
        if match[2].start < fill[0][2].stop and match[2].stop > fill[0][2].start:
            overlap[fill] = None
    max_score = 0
    for o in overlap:
        if o[1] > max_score:
            max_score = o[1]
    if max_score < 3:
        find_match_score = match[4]
        if not find_match_score:
            find_match_score = find_match(tokens, match, pub_titles, biotools, ids_pub, find_url_tokens)
        if not overlap or find_match_score > max_score:
            for o in overlap:
                del results[o[0][2][0]]
                filled.discard(o)
            results[match[2][0]] = match
            filled.add((match, find_match_score))
            for o in overlap:
                matches_unused[o[0]] = None
            for match_unused in list(matches_unused.keys()):
                fill_result(match_unused, results, filled, matches_unused, tokens, pub_titles, biotools, ids_pub, find_url_tokens)
        else:
            matches_unused[match[:-1] + (find_match_score,)] = None
    else:
        matches_unused[match] = None


def fill_results(matches, results, filled, matches_unused, tokens, pub_titles, biotools, ids_pub, find_url_tokens):
    for match in sorted(matches, key=lambda x: (not x[3], len(x[2]), len(x[0]), -len(x[1]), x[1], x[2][0], x[0], x[4]), reverse=True):
        fill_result(match, results, filled, matches_unused, tokens, pub_titles, biotools, ids_pub, find_url_tokens)


def get_results(tokens, biotools, cites, pub_titles, counts, wordlist, soup):
    matches = get_matches(tokens, biotools)

    ids_pub = get_ids_pub(soup)
    find_url_tokens = {}
    for token_label in Tokens:
        find_url_tokens[token_label] = get_find_url_tokens(tokens[token_label])
    matches_filtered = {}
    for token_label in Tokens:
        matches_filtered[token_label] = filter_matches(tokens[token_label], matches[token_label], cites, pub_titles, counts[token_label], wordlist, biotools, ids_pub, find_url_tokens[token_label])

    results = {}
    filled = set()
    matches_unused = {}
    for token_label in Tokens:
        fill_results(matches_filtered[token_label], results, filled, matches_unused, tokens[token_label], pub_titles, biotools, ids_pub, find_url_tokens[token_label])

    return results


def percent_encode(text):
    return ''.join('%{:02x}'.format(ord(c)) for c in text)


def html_encode(text):
    no_encode = {38, 35, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59} # &#[0-9];
    encoded = ''.join(c if ord(c) in no_encode else f'&#{ord(c)};' for c in text)
    double_escaped = {'&&#108;&#116;;': '&lt;', '&&#103;&#116;;': '&gt;', '&&#97;&#109;&#112;;': '&amp;'}
    for k, v in double_escaped.items():
        encoded = encoded.replace(k, v)
    return encoded


def html_encode_unicode(text):
    escape = {'<': '&lt;', '>': '&gt;', '&': '&amp;'}
    return ''.join(escape[c] if c in escape else c if 32 <= ord(c) < 127 else f'&#{ord(c)};' for c in text)


def html_replace(id, biotools, color):
    def replace(match):
        match_begin_end = re.match(r'^(\W*)(.*?)([^\w)+]*)$', unescape(match.group(1)), flags=re.DOTALL)
        begin = match_begin_end.group(1)
        name = match_begin_end.group(2)
        end = match_begin_end.group(3)
        if name.endswith(')') and (begin.endswith('(') or '(' not in name):
            name = name[:-1]
            end = ')' + end
        if name.endswith('+') and not biotools[id]['name'].endswith('+'):
            name = name[:-1]
            end = '+' + end
        return rf'{html_encode_unicode(begin)}<a href="https://{percent_encode('bio.tools')}/{percent_encode(id)}" style="background-color: {color}; padding: 2px; font-weight: bold;">{html_encode(name)}</a>{html_encode_unicode(end)}'
    return replace


def html_links_in_title(html, link_tag):
    title_content = re.search(rf'<{link_tag}>(.*?)</{link_tag}>', html, flags=re.DOTALL)
    if title_content:
        return len(re.findall(r'</a>', title_content.group(1)))
    return 0


def html_warning(html, message):
    return html.replace('<body>', f'<body>\n<p style="color: #ff0000; font-weight: bold;">{html_encode(message)}</p>')


def get_html(results, tokens_orig, pub_titles, biotools, soup):
    xml_doc = etree.fromstring(str(soup).encode('utf-8'))
    with open(get_file_path('jats.xslt'), 'r', encoding='utf-8') as file:
        xslt_doc = etree.parse(file)
    transform = etree.XSLT(xslt_doc)
    title_text = None
    try:
        result_tree = transform(xml_doc)
        if result_tree.xpath('//title') and result_tree.xpath('//title')[0] is not None:
            title_text = result_tree.xpath('//title')[0].text
        html = etree.tostring(result_tree, pretty_print=True, method="html", doctype='<!DOCTYPE html>').decode("utf-8")
    except etree.XSLTApplyError:
        html = '<!DOCTYPE html>\n<html><head><title></title></head><body>\n' + soup2text(soup) + '\n</body></html>\n'

    id_tokens = {}
    id_count = {}
    for pos, match in sorted(results.items()):
        token = ' '.join(tokens_orig[i] for i in match[2])
        if match[1] in id_tokens:
            id_tokens[match[1]].add(token)
            id_count[match[1]] += 1
        else:
            id_tokens[match[1]] = {token}
            id_count[match[1]] = 1
    to_front = []
    keys = list(id_tokens.keys())
    for i in range(len(keys)):
        tokens_i = id_tokens[keys[i]]
        for j in range(i + 1, len(keys)):
            tokens_j = id_tokens[keys[j]]
            contains = False
            for token_i in tokens_i:
                for token_j in tokens_j:
                    if token_i in token_j:
                        contains = True
                        if keys[j] in to_front:
                            to_front.remove(keys[j])
                        to_front.append(keys[j])
                        break
                if contains:
                    break
    for key in to_front:
        value = id_tokens.pop(key)
        id_tokens_new = {key: value}
        id_tokens_new.update(id_tokens)
        id_tokens = id_tokens_new
    links_in_title_total = 0
    links_in_h1_total = 0
    colors = ["#FFFF00", "#FFCC66", "#FF0000", "#FFFF66", "#FF9933", "#FF6666", "#FFFFCC", "#FF6600", "#FFCCCC", "#FFFF33", "#FFCC33", "#FF3333", "#FFFF99", "#FF9900", "#FF9999"]
    color_index = 0
    for id, tokens in id_tokens.items():
        color = colors[color_index % len(colors)]
        color_index += 1
        pattern = r'(?<=[\s/>])(' # TODO en-dash, em-dash
        pattern += r'|'.join([r'[\s/]+'.join([re.escape(html_encode_unicode(token_word)) for token_word in token.split()]) for token in sorted(tokens, key=lambda x: (len(x), x), reverse=True)])
        pattern += r')(?=[\s/<])' # TODO \s does not match all Unicode whitespace characters, like Unicode thin space (&#8201;)
        html, n = re.compile(pattern).subn(html_replace(id, biotools, color), html)
        links_in_title = html_links_in_title(html, 'title') - links_in_title_total
        links_in_h1 = html_links_in_title(html, 'h1') - links_in_h1_total
        n -= (links_in_title + links_in_h1)
        links_in_title_total += links_in_title
        links_in_h1_total += links_in_h1
        if n != id_count[id]:
            html = html_warning(html, f'Number of highlighted matches ({n}) does not equal number of found matches ({id_count[id]}) for pattern {pattern}')
    if title_text:
        html = re.sub(r'<title>.*?</title>', f'<title>{title_text}</title>', html, count=1, flags=re.DOTALL)

    ref_ids = get_ref_ids(pub_titles, biotools, get_ids_pub(soup))
    unmatched_refs = [key for key in ref_ids.keys() if key in ref_ids.keys() - id_tokens.keys() and (ref_ids[key] not in EXCLUDED_CITES or key in EXCLUDED_CITES_EXCEPTION_IDS)]
    if unmatched_refs:
        unmatched = ', '.join(f'{biotools[id]['name']} [{id}] ({ref_ids[id]})' for id in unmatched_refs)
        html = html_warning(html, f'Unmatched bio.tools in References (cited but not mentioned): {unmatched}')
        # TODO not entirely correct as mentions in references are also matched
    uncited_ids = [key for key in id_tokens.keys() if key in id_tokens.keys() - ref_ids.keys()]
    if uncited_ids:
        uncited = ', '.join(f'{biotools[id]['name']} [{id}]' for id in uncited_ids)
        html = html_warning(html, f'Matched bio.tools not in References (mentioned but not cited): {uncited}')
    # TODO could also output warning if homepage (or other link?) found but no mention matches

    return html


def soup2text(soup):
    front_text = soup.find('front').get_text(separator=' ') if soup.find('front') else ''
    body_text = soup.find('body').get_text(separator=' ') if soup.find('body') else ''
    back_text = soup.find('back').get_text(separator=' ') if soup.find('back') else ''
    return '\n\n'.join(filter(None, [front_text, body_text, back_text]))


def get_xml_europepmc(pmc):
    response = requests.get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML')
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'lxml-xml')
    return soup2text(soup), soup


def get_file_path(file):
    return str(os.path.join(os.path.dirname(os.path.abspath(__file__)), file))


def get_biotools():
    biotoolsdump = get_file_path('../biotoolspub/biotoolsdump.zip')
    biotools = load_biotools_from_zip(biotoolsdump, 'biotools.json')
    return {tool['biotoolsID']: tool for tool in biotools}


def match_xml(pmc: str | list[str]) -> str:
    biotools = get_biotools()

    cites = read_cites_counts(get_file_path(f'cites.tsv'), True)
    pub_titles = {k: (v, normalize_pub_title(v)) for k, v in read_cites_counts(get_file_path(f'pub_titles.tsv'), False).items()}
    counts = {}
    for token_label in Tokens:
        counts[token_label] = read_cites_counts(get_file_path(f'counts_{token_label.value}.tsv'), True)

    wordlist = set()
    with open(get_file_path('wamerican-wbritish-insane'), 'r', encoding='utf-8') as file:
        for line in file:
            wordlist.add(line.strip().lower())

    html = ''
    if isinstance(pmc, str):
        pmc = [pmc]
    for p in pmc:
        text, soup = get_xml_europepmc(p)
        tokens_orig = get_tokens(text)
        tokens = {Tokens.CASED: get_tokens_cased(tokens_orig), Tokens.LOWER: get_tokens_lower(tokens_orig)}

        results = get_results(tokens, biotools, cites, pub_titles, counts, wordlist, soup)

        html_p = get_html(results, tokens_orig, pub_titles, biotools, soup)
        html_p = html_p.replace('<body>', f'<body>\n<p>{p}</p>')
        html += html_p
    html = re.sub(r'</body>.*?<body>', '<hr>', html, flags=re.DOTALL)
    html = re.sub(r'<title>.*?</title>', f'<title>{', '.join(pmc)}</title>', html, count=1, flags=re.DOTALL)

    # with open('output.html', 'w', encoding='utf-8', newline='') as f:
    #     f.write(html)

    return html


def get_counts(counts, matches):
    matches_done = set()
    for s in matches:
        if s[3]:
            continue
        if s[0] in counts:
            counts[s[0]] += 1 if s[0] not in matches_done else 0
        else:
            counts[s[0]] = 1
        matches_done.add(s[0])


def write_cites_counts(file_path, cites_counts, cites_counts_col_a, cites_counts_col_b):
    cites_counts = dict(sorted(cites_counts.items(), key=lambda x: (-x[1], x[0]) if type(x[1]) == int else x[0]))
    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t', lineterminator='\n')
        writer.writerow([cites_counts_col_a, cites_counts_col_b])
        for key, value in cites_counts.items():
            writer.writerow([key, value])


def read_cites_counts(file_path, ints):
    cites_counts = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for row in reader:
            cites_counts[row[0]] = int(row[1]) if ints else row[1]
    return cites_counts


def generate_counts():
    biotools = get_biotools()
    cites = {}
    pub_titles = {}
    i = 0
    for tool in biotools.values():
        cites[tool['biotoolsID']] = -1
        # TODO possibly exclude publication type "Other"
        for publication in tool['publication']:
            found = False
            for _ in range(3):
                for id_label in 'pmid', 'pmcid', 'doi':
                    if publication[id_label]:
                        if id_label == 'doi':
                            id = normalize_doi(publication[id_label])
                        else:
                            id = publication[id_label].strip().lower()
                        params = {
                            'query': f'ext_id:{id} src:med' if id_label == 'pmid' else f'pmcid:{id}' if id_label == 'pmcid' else f'doi:{id}'
                        }
                        response = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params=params)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.content, 'lxml-xml')
                            title = soup.find('title')
                            if title:
                                title = title.text
                                for pub_title_label in 'pmid', 'pmcid', 'doi':
                                    label = soup.find(pub_title_label)
                                    if label:
                                        if pub_title_label == 'doi':
                                            pub_titles[normalize_doi(label.text)] = title
                                        else:
                                            pub_titles[label.text.strip().lower()] = title
                            citedByCount = soup.find('citedByCount')
                            if citedByCount:
                                if cites[tool['biotoolsID']] < 0:
                                    cites[tool['biotoolsID']] = 0
                                cites[tool['biotoolsID']] += int(citedByCount.text)
                                found = True
                                break
                        else:
                            print(response.status_code, flush=True)
                            time.sleep(1)
                if found:
                    break
        i += 1
        print(f'cites {i}', flush=True)
        time.sleep(0.1)
    write_cites_counts(get_file_path(f'cites.tsv'), cites, 'biotoolsID', 'Cites')
    write_cites_counts(get_file_path(f'pub_titles.tsv'), pub_titles, 'Publication', 'Title')

    fulltexts = get_file_path('../biotoolspub/fulltexts')
    counts = {}
    for token_label in Tokens:
        counts[token_label] = {}
    i = 0
    for xml in os.listdir(fulltexts):
        with open(os.path.join(fulltexts, xml), 'r', encoding='utf-8') as file:
            soup = BeautifulSoup('<article>' + file.read() + '</article>', 'xml')
            text = soup.get_text(separator=' ')
        tokens_orig = get_tokens(text)
        tokens = {Tokens.CASED: get_tokens_cased(tokens_orig), Tokens.LOWER: get_tokens_lower(tokens_orig)}
        matches = get_matches(tokens, biotools)
        for token_label in Tokens:
            get_counts(counts[token_label], matches[token_label])
        i += 1
        print(f'counts {i}', flush=True)
    for token_label in Tokens:
        counts[token_label][''] = i
        write_cites_counts(get_file_path(f'counts_{token_label.value}.tsv'), counts[token_label], 'Match', 'Counts')


def main():
    # match_xml([
    #     'PMC3257301', 'PMC10845142', 'PMC11387482', 'PMC1160178', 'PMC11330317',
    #     'PMC3125781', 'PMC7005830', 'PMC4422463', # comet
    #     'PMC6612819', 'PMC8479664', 'PMC7699628', 'PMC4086120', 'PMC5570231', # prism
    #     'PMC29831', 'PMC126239', 'PMC540004', 'PMC1160116', 'PMC1538874', 'PMC2238880', 'PMC3245168', 'PMC5154471',
    #     'PMC7174305', 'PMC9500060', 'PMC10578359', 'PMC11301774',
    #     'PMC13900', 'PMC55319', 'PMC100320', 'PMC260744', 'PMC411032', 'PMC1033552', 'PMC1190152', 'PMC1342034', 'PMC1490295', 'PMC1643837',
    #     'PMC1810238', 'PMC1961630', 'PMC2110001', 'PMC2261226', 'PMC2410013', 'PMC2560510', 'PMC2710001', 'PMC2860042', 'PMC3010001', 'PMC3160001',
    #     'PMC3310001', 'PMC3460001', 'PMC3610010', 'PMC3760028', 'PMC3910002', 'PMC4060003', 'PMC4210052', 'PMC4360001', 'PMC4510001', 'PMC4660001',
    #     'PMC4810001', 'PMC4960005', 'PMC5110002', 'PMC5260001', 'PMC5410001', 'PMC5560001', 'PMC5710014', 'PMC5860004', 'PMC6010001', 'PMC6160001',
    #     'PMC6310011', 'PMC6460013', 'PMC6610001', 'PMC6760002', 'PMC6910015', 'PMC7060019', 'PMC7210001', 'PMC7360001', 'PMC7510001', 'PMC7680004',
    #     'PMC7830001', 'PMC7980034', 'PMC8130002', 'PMC8280001', 'PMC8430027', 'PMC8580007', 'PMC8730001', 'PMC8880001', 'PMC9030001', 'PMC9180001',
    #     'PMC9330001', 'PMC9480001', 'PMC9630018', 'PMC9780001', 'PMC9930001', 'PMC10080001', 'PMC10230001', 'PMC10380001', 'PMC10530004', 'PMC10680001',
    #     'PMC10830001', 'PMC10980001', 'PMC11130001', 'PMC11280001', 'PMC11430001'
    # ])
    # match_xml('PMC2238880')
    # generate_counts()
    pass


if __name__ == '__main__':
    main()
