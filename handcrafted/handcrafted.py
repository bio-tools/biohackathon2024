import csv
import os
import re
import unicodedata
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from lxml import etree

from bh24_literature_mining.utils import load_biotools_from_zip


MAX_WORDS = 8


def get_tokens(text):
    text = text.replace('/', ' ') # en-dash, em-dash
    return text.split()


def get_tokens_cased(tokens):
    return [''.join(char for char in token if char.isalnum()) for token in tokens]


def lower(token):
    return ''.join(char for char in unicodedata.normalize('NFD', token) if char.isalpha()).lower()


def get_tokens_lower(tokens):
    return [lower(token) for token in tokens]


def get_conc_list(tokens, n):
    conc_list = []
    for i in range(0, len(tokens) - n):
        token = ' '.join(tokens[j] for j in range(i, i + n + 1))
        conc_list.append(token)
    return conc_list


def get_matches_tool(s, tool, tokens_orig_set, tokens_cased_set, tokens_lower_set, matches_orig, matches_cased, matches_lower):
    tool_orig = s.split()
    tool_cased = get_tokens_cased(tool_orig)
    tool_lower = get_tokens_lower(tool_orig)
    tool_orig_str = ' '.join(tool_orig)
    tool_cased_str = ' '.join(tool_cased)
    tool_lower_str = ' '.join(tool_lower)
    n = len(tool_orig) - 1
    if n < MAX_WORDS:
        if tool_orig_str in tokens_orig_set[n]:
            matches_orig.add((tool_orig_str, tool['biotoolsID']))
        if tool_cased_str in tokens_cased_set[n]:
            matches_cased.add((tool_cased_str, tool['biotoolsID']))
        if tool_lower_str in tokens_lower_set[n]:
            matches_lower.add((tool_lower_str, tool['biotoolsID']))


def get_matches(tokens_orig, tokens_cased, tokens_lower, biotools):
    tokens_orig_set = []
    tokens_cased_set = []
    tokens_lower_set = []
    for i in range(0, MAX_WORDS):
        tokens_orig_set.append(set(get_conc_list(tokens_orig, i)))
        tokens_cased_set.append(set(get_conc_list(tokens_cased, i)))
        tokens_lower_set.append(set(get_conc_list(tokens_lower, i)))

    matches_orig: set[(str, str)] = set()
    matches_cased: set[(str, str)] = set()
    matches_lower: set[(str, str)] = set()
    for tool in biotools.values():
        get_matches_tool(tool['name'].strip(), tool, tokens_orig_set, tokens_cased_set, tokens_lower_set, matches_orig, matches_cased, matches_lower)
        get_matches_tool(tool['biotoolsID'], tool, tokens_orig_set, tokens_cased_set, tokens_lower_set, matches_orig, matches_cased, matches_lower)
        # TODO compound word matching ("x y" to "xy" and "xy" to "x y")
        # TODO approximate matching (one insertion, deletion, substitution, transposition)
        # TODO only some words of a multi-word tool name (like acronym in parentheses)
        # TODO synonyms (from SciCrunch API)
        # TODO construct acronyms

    return matches_orig, matches_cased, matches_lower


def find_url(tokens, match, biotools):
    found = False
    tool = biotools[match[1]]
    urls_parsed = set()
    urls_parsed.add(urlparse(tool['homepage']))
    for link_label in ['link', 'documentation', 'download']:
        for link in tool[link_label]:
            if 'Software catalogue' not in link['type']:
                urls_parsed.add(urlparse(link['url']))
    for url_parsed in urls_parsed:
        domain = ''.join(char for char in url_parsed.netloc if char.isalnum()).lower()
        paths = [path.lower() for path in url_parsed.path.split('/')[1:]]
        if domain.startswith('www'):
            domain = domain[3:]
        for i in range(len(tokens)):
            token_lower = tokens[i].lower()
            if token_lower.startswith('www'):
                token_lower = token_lower[3:]
            if domain == token_lower:
                found = True
                if len(paths) > 0 and i + 1 < len(tokens):
                    if paths[0] != tokens[i + 1].lower():
                        found = False
                if len(paths) > 1 and i + 2 < len(tokens):
                    if paths[1] != tokens[i + 2].lower():
                        found = False
                if not found:
                    match_lower = match[0].lower()
                    for j in range(i + 1, i + 1 + 2):
                        if j < len(tokens):
                            token_lower = tokens[j].lower()
                            if match_lower == token_lower or len(match_lower) > 2 and match_lower in token_lower:
                                found = True
                                break
                if found:
                    break
        if found:
            break
    return found


def normalize_doi(doi):
    doi = doi.strip().lower()
    for prefix in 'doi:', 'http://doi.org/', 'https://doi.org/', 'http://dx.doi.org/', 'https://dx.doi.org/':
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi


def find_ref(match, biotools, soup):
    found = False
    tool = biotools[match[1]]
    ids = {}
    id_labels = ['pmid', 'pmcid', 'doi']
    for id_label in id_labels:
        ids[id_label] = set()
    for publication in tool['publication']:
        for id_label in id_labels:
            if publication[id_label]:
                if id_label == 'doi':
                    ids[id_label].add(normalize_doi(publication[id_label]))
                else:
                    ids[id_label].add(publication[id_label].strip().lower())
    for id_label, v in ids.items():
        ids_pub = set()
        ids_pub.update([tag.text.strip().lower() for tag in soup.find_all('pub-id', {'pub-id-type': id_label})])
        ids_pub.update([tag.text.strip().lower() for tag in soup.find_all('article-id', {'pub-id-type': id_label})])
        if id_label == 'doi':
            ids_pub = {normalize_doi(id_pub) for id_pub in ids_pub}
        for id in v:
            for id_pub in ids_pub:
                if id == id_pub:
                    found = True
                    break
            if found:
                break
        if found:
            break
    return found


def filter_match(tokens, match, biotools, soup):
    if find_url(tokens, match, biotools):
        return True
    if find_ref(match, biotools, soup):
        return True
    return False


def filter_matches(tokens, matches, scores, wamerican_insane, biotools, soup):
    matches_filtered: set[(str, str)] = set()
    for match in matches:
        passes = False
        # TODO tune algorithm
        if not match[0] in scores or scores[match[0]] <= 5:
            if len([c for c in match[0][1:] if not c.islower() and c != ' ']) > 0:
                passes = True
            else:
                for word in match[0].split():
                    if lower(word) not in wamerican_insane:
                        passes = True
        if passes or filter_match(tokens, match, biotools, soup):
            matches_filtered.add(match)
    return matches_filtered


def attach_matches(results, tokens, matches, matches_label):
    for match in matches:
        for i in range(len(tokens)):
            matching = True
            j = 0
            for m in match[0].split():
                if i + j >= len(tokens) or m != tokens[i + j]:
                    matching = False
                    break
                j += 1
            if matching:
                if i in results:
                    results[i].append((match[0], match[1], matches_label))
                else:
                    results[i] = [(match[0], match[1], matches_label)]


def get_xml_europepmc(pmc):
    response = requests.get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML')
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'lxml-xml')
    return soup.get_text(separator=' '), soup


def get_file_path(file):
    return str(os.path.join(os.path.dirname(os.path.abspath(__file__)), file))


def get_biotools():
    biotoolsdump = get_file_path('../biotoolspub/biotoolsdump.zip')
    biotools = load_biotools_from_zip(biotoolsdump, 'biotools.json')
    return {tool['biotoolsID']: tool for tool in biotools}


def get_results(tokens_orig, tokens_cased, tokens_lower, soup):
    biotools = get_biotools()

    matches_orig, matches_cased, matches_lower = get_matches(tokens_orig, tokens_cased, tokens_lower, biotools)

    scores_orig = get_scores(get_file_path('counts_orig.tsv'))
    scores_cased = get_scores(get_file_path('counts_cased.tsv'))
    scores_lower = get_scores(get_file_path('counts_lower.tsv'))

    wamerican_insane = set()
    with open(get_file_path('wamerican-insane'), 'r') as file:
        for line in file:
            wamerican_insane.add(lower(line.strip()))

    matches_orig = filter_matches(tokens_orig, matches_orig, scores_orig, wamerican_insane, biotools, soup)
    matches_cased = filter_matches(tokens_cased, matches_cased, scores_cased, wamerican_insane, biotools, soup)
    matches_lower = filter_matches(tokens_lower, matches_lower, scores_lower, wamerican_insane, biotools, soup)

    results_init = {}
    attach_matches(results_init, tokens_orig, matches_orig, 'orig')
    attach_matches(results_init, tokens_cased, matches_cased, 'cased')
    attach_matches(results_init, tokens_lower, matches_lower, 'lower')
    results_init = dict(sorted(results_init.items()))
    results = {}
    for pos, matches in results_init.items():
        ids = set()
        for match in matches:
            ids.add(match[1])
        if len(ids) > 1:
            found = False
            for match in matches:
                if filter_match(tokens_orig if match[2] == 'orig' else tokens_cased if match[2] == 'cased' else tokens_lower, (match[0], match[1]), biotools, soup):
                    results[pos] = match[1], len(match[0].split())
                    found = True
                    break
            if not found:
                results[pos] = matches[0][1], len(matches[0][0].split())
        else:
            results[pos] = matches[0][1], len(matches[0][0].split())
    # TODO in case of conflicts the match is chosen based on order orig, cased, lower, but there can also be conflicts within them, then first within them is chosen (random)
    # TODO currently multiword match can have a different match within
    return results


def get_html(results, tokens_orig, soup):
    xml_doc = etree.fromstring(str(soup).encode('utf-8'))
    with open(get_file_path('jats.xslt'), 'rb') as file:
        xslt_doc = etree.parse(file)
    transform = etree.XSLT(xslt_doc)
    result_tree = transform(xml_doc)
    html = etree.tostring(result_tree, pretty_print=True, method="html", doctype='<!DOCTYPE html>').decode("utf-8")

    id_tokens = {}
    for pos, id in results.items():
        token = ''
        for i in range(id[1]):
            token += tokens_orig[pos + i] + ' '
        token = token.strip()
        if id[0] in id_tokens:
            id_tokens[id[0]].add(token)
        else:
            id_tokens[id[0]] = {token}
    for id, tokens in id_tokens.items():
        pattern = r'([\s/>])('
        pattern += r'|'.join([r'\s+'.join([re.escape(token_word) for token_word in token.split()]) for token in tokens])
        pattern += r')([\s/<])'
        html, n = re.compile(pattern).subn(rf'\1<a href="https://bio.tools/{''.join('%{:02x}'.format(ord(c)) for c in id)}" style="background-color: yellow; padding: 2px; font-weight: bold;">\2</a>\3', html)
        # TODO should actually percent encode the text between <a>
        # TODO check with n if any matches not found (output warning to html that failed to highlight)
        # TODO could also output warning that citation of tool found but no matches were found (same for homepage and other links)
        # TODO could also output warning if any conflicts remained for a match position
        # TODO could use different color for different matches
        # TODO remove extra symbols from highlight (begin and end), check what symbols can there be at all from bio.tools "name"

    return html


def match_xml(pmc):
    text, soup = get_xml_europepmc(pmc)
    tokens_orig = get_tokens(text)
    tokens_cased = get_tokens_cased(tokens_orig)
    tokens_lower = get_tokens_lower(tokens_orig)

    results = get_results(tokens_orig, tokens_cased, tokens_lower, soup)
    html = get_html(results, tokens_orig, soup)

    # with open('output.html', 'w', encoding='utf-8') as f:
    #     f.write(html)

    return html


def get_counts(counts, matches, cites):
    matches_done = set()
    for s in matches:
        if s[0] in counts:
            counts_0 = counts[s[0]][0] + (1 if s[0] not in matches_done else 0)
            counts_1 = counts[s[0]][1] + (cites[s[1]] if s[1] not in counts[s[0]][2] else 0)
            counts[s[0]][2].add(s[1])
            counts[s[0]] = counts_0, counts_1, counts[s[0]][2]
        else:
            counts[s[0]] = 1, cites[s[1]], {s[1]}
        matches_done.add(s[0])


def sort_counts(counts):
    return dict(sorted(counts.items(), key=lambda x: x[1][0], reverse=True))


def write_counts(file_path, counts):
    with open(file_path, 'w', newline='\n') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['Match', 'Count', 'Cites'])
        for key, value in counts.items():
            writer.writerow([key, value[0], value[1]])


def get_scores(file_path):
    scores = {}
    with open(file_path, 'r', newline='\n') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for row in reader:
            if int(row[2]) > 0:
                scores[row[0]] = int(row[1]) / int(row[2])
            else:
                scores[row[0]] = int(row[1]) + 1
    return scores


def generate_counts():
    biotools = get_biotools()
    cites = {}
    i = 0
    for tool in biotools.values():
        cites[tool['biotoolsID']] = 0
        # TODO should all publications be taken or only Primary?
        for publication in tool['publication']:
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
                        citedByCount = soup.find('citedByCount')
                        if citedByCount:
                            cites[tool['biotoolsID']] += int(citedByCount.text)
                            break
        i += 1
        print(f'cites {i}')
    fulltexts = get_file_path('../biotoolspub/fulltexts')
    counts_orig = {}
    counts_cased = {}
    counts_lower = {}
    i = 0
    for xml in os.listdir(fulltexts):
        with open(os.path.join(fulltexts, xml), 'r', encoding='utf-8') as file:
            soup = BeautifulSoup('<article>' + file.read() + '</article>', 'xml')
            text = soup.get_text(separator=' ')
        tokens_orig = get_tokens(text)
        tokens_cased = get_tokens_cased(tokens_orig)
        tokens_lower = get_tokens_lower(tokens_orig)
        matches_orig, matches_cased, matches_lower = get_matches(tokens_orig, tokens_cased, tokens_lower, biotools)
        get_counts(counts_orig, matches_orig, cites)
        get_counts(counts_cased, matches_cased, cites)
        get_counts(counts_lower, matches_lower, cites)
        i += 1
        print(f'counts {i}')
    counts_orig = sort_counts(counts_orig)
    counts_cased = sort_counts(counts_cased)
    counts_lower = sort_counts(counts_lower)
    write_counts(get_file_path('counts_orig.tsv'), counts_orig)
    write_counts(get_file_path('counts_cased.tsv'), counts_cased)
    write_counts(get_file_path('counts_lower.tsv'), counts_lower)


def main():
    #match_xml('PMC3257301')
    #generate_counts()
    pass


if __name__ == '__main__':
    main()
