import csv
import os
import re
import time
import unicodedata
from enum import Enum
from urllib.parse import urlparse
from html import unescape

import requests
from bs4 import BeautifulSoup
from lxml import etree

from bh24_literature_mining.utils import load_biotools_from_zip


MAX_WORDS = 8
MAX_WORDS_COMPOUND = 4

EXCLUDED_ACRONYMS_IN_TOOL_NAME = {'EBI', 'NIH', 'SIB'}
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
EXCLUDED_CITES = {
    '25633503', 'pmc4509590', '10.1038/nmeth.3252', # Bioconductor
    '27137889', 'pmc4987906', '10.1093/nar/gkw343', # Galaxy
    '10.7490/f1000research.1114334.1', # Galaxy Pasteur
    '10827456', '10.1016/s0168-9525(00)02024-2', # EMBOSS
    '10.1017/cbo9781139151405', # EMBOSS
    '10.1017/cbo9781139151399', # EMBOSS
    '35412617', 'pmc9252731', '10.1093/nar/gkac240', # EMBL-EBI
    '26673705', 'pmc4702932', '10.1093/nar/gkv1352', # EMBL-EBI
    '26687719', 'pmc4702834', '10.1093/nar/gkv1157' # Ensembl
}
EXCLUDED_CITES_EXCEPTION_IDS = {'bioconductor', 'galaxy', 'emboss', 'ebi_tools', 'ensembl'}


class Tokens(Enum):
    ORIG = 'orig'
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
    tool = {Tokens.ORIG: name.split()}
    tool[Tokens.CASED] = get_tokens_cased(tool[Tokens.ORIG])
    tool[Tokens.LOWER] = get_tokens_lower(tool[Tokens.ORIG])
    tool_str = {}
    for token_label in Tokens:
        tool_str[token_label] = {' '.join([t for t in tool[token_label] if t])}
    n = len(tool[Tokens.ORIG]) - 1
    if n < MAX_WORDS_COMPOUND:
        for token_label in Tokens:
            for i in range(1, n + 1):
                tool_str[token_label].add(' '.join(tool[token_label][j] for j in range(0, i) if tool[token_label][j]) + ' '.join(tool[token_label][j] for j in range(i, n + 1) if tool[token_label][j]))
    for token_label in Tokens:
        for i in range(0, MAX_WORDS):
            for s in tool_str[token_label] & tokens_set[token_label][i]:
                for w in tokens_words[token_label][i]:
                    if s == w[0]:
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
    return ids_pub


def find_ref(match, biotools, ids_pub):
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
        for id in v:
            if id in EXCLUDED_CITES and match[1] not in EXCLUDED_CITES_EXCEPTION_IDS:
                continue
            if id in ids_pub[id_label]:
                found = True
                break
        if found:
            break
    return found


def get_ref_ids(biotools, ids_pub):
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
                    if id in ids_pub[id_label]:
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
    path = path.split('?')[0]
    path = path.removesuffix('html')
    path = path.removesuffix('htm')
    path = path.removesuffix('.')
    return path


def find_url(tokens, match, biotools, find_url_tokens):
    found = False
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
        if domain.startswith('www.'):
            domain = domain[4:]
        if domain.startswith('www'):
            domain = domain[3:]
        for i in range(len(find_url_tokens)):
            token_lower = find_url_tokens[i]
            if domain == token_lower:
                found = True
                paths_seen = 0
                match_lower = match[0].lower()
                for j in range(len(paths)):
                    if i + 1 + j == len(tokens):
                        break
                    if paths[j] not in SKIP_PATHS:
                        paths_seen += 1
                        token_lower = tokens[i + 1 + j].lower()
                        token_lower_normalized = normalize_path(token_lower)
                        if match_lower == token_lower or match_lower == token_lower_normalized:
                            found = True
                            break
                        if paths[j] != token_lower and normalize_path(paths[j]) != token_lower_normalized:
                            found = False
                        if paths_seen == 2:
                            break
                if found:
                    break
        if found:
            break
    return found


def find_match(tokens, match, biotools, ids_pub, find_url_tokens):
    if match[0] in EXCLUDED_MATCHES:
        return False
    if find_ref(match, biotools, ids_pub):
        return True
    if find_url(tokens, match, biotools, find_url_tokens):
        return True
    return False


def filter_matches(tokens, matches, scores, wordlist, biotools, ids_pub, find_url_tokens):
    matches_filtered: set[(str, str, list[int], bool)] = set()
    matches_pass: set[(str, str, bool)] = set()
    matches_fail: set[(str, str, bool)] = set()
    for match in matches:
        if (match[0], match[1], match[3]) in matches_pass:
            matches_filtered.add(match)
            continue
        if (match[0], match[1], match[3]) in matches_fail:
            continue
        passes = False
        if not match[3]:
            score = scores[match[0]] if match[0] in scores else 1
            simple = match[0].isalpha() and (match[0].islower() or match[0].isupper())
            if simple and score <= 0.2 or not simple and score <= 5:
                if len([c for c in match[0][1:] if not c.islower() and c != ' ']) > 0:
                    passes = True
                else:
                    for word in match[0].split():
                        if lower(word) not in wordlist:
                            passes = True
        if passes or find_match(tokens, match, biotools, ids_pub, find_url_tokens):
            matches_filtered.add(match)
            matches_pass.add((match[0], match[1], match[3]))
        else:
            matches_fail.add((match[0], match[1], match[3]))
    return matches_filtered


def fill_results(matches, results, filled, tokens, biotools, ids_pub, find_url_tokens):
    for match in sorted(matches, key=lambda x: (not x[3], len(x[2]), len(x[0]), -len(x[1]), x[1]), reverse=True):
        overlap = set()
        for fill in filled:
            if match[2].start < fill[0].stop and match[2].stop > fill[0].start:
                overlap.add(fill)
        overwrite = True
        for o in overlap:
            if o[1]:
                overwrite = False
                break
        if overwrite:
            find = False # find = match[3]
            if not find:
                find = find_match(tokens, match, biotools, ids_pub, find_url_tokens)
            if not overlap or find:
                for o in overlap:
                    del results[o[0][0]]
                filled -= overlap
                results[match[2][0]] = match[1], len(match[2])
                filled.add((match[2], find))


def get_results(tokens, biotools, scores, wordlist, soup):
    matches = get_matches(tokens, biotools)

    ids_pub = get_ids_pub(soup)
    find_url_tokens = {}
    for token_label in Tokens:
        find_url_tokens[token_label] = get_find_url_tokens(tokens[token_label])
    matches_filtered = {}
    for token_label in Tokens:
        matches_filtered[token_label] = filter_matches(tokens[token_label], matches[token_label], scores[token_label], wordlist, biotools, ids_pub, find_url_tokens[token_label])

    results = {}
    filled = set()
    for token_label in Tokens:
        fill_results(matches_filtered[token_label], results, filled, tokens[token_label], biotools, ids_pub, find_url_tokens[token_label])

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
        if name.endswith(')') and (begin.endswith('(') or '(' not in name or not biotools[id]['name'].endswith(')')):
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


def get_html(results, tokens_orig, biotools, soup):
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
    for pos, id in sorted(results.items()):
        token = ' '.join(tokens_orig[pos + i] for i in range(id[1]))
        if id[0] in id_tokens:
            id_tokens[id[0]].add(token)
            id_count[id[0]] += 1
        else:
            id_tokens[id[0]] = {token}
            id_count[id[0]] = 1
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
        pattern += r'|'.join([r'[\s/]+'.join([re.escape(html_encode_unicode(token_word)) for token_word in token.split()]) for token in tokens])
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

    ref_ids = get_ref_ids(biotools, get_ids_pub(soup))
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

    scores = {}
    for token_label in Tokens:
        scores[token_label] = get_scores(get_file_path(f'counts_{token_label.value}.tsv'))

    wordlist = set()
    with open(get_file_path('wamerican-wbritish-insane'), 'r', encoding='utf-8') as file:
        for line in file:
            wordlist.add(lower(line.strip()))

    html = ''
    if isinstance(pmc, str):
        pmc = [pmc]
    for p in pmc:
        text, soup = get_xml_europepmc(p)
        tokens = {Tokens.ORIG: get_tokens(text)}
        tokens[Tokens.CASED] = get_tokens_cased(tokens[Tokens.ORIG])
        tokens[Tokens.LOWER] = get_tokens_lower(tokens[Tokens.ORIG])

        results = get_results(tokens, biotools, scores, wordlist, soup)

        html_p = get_html(results, tokens[Tokens.ORIG], biotools, soup)
        html_p = html_p.replace('<body>', f'<body>\n<p>{p}</p>')
        html += html_p
    html = re.sub(r'</body>.*?<body>', '<hr>', html, flags=re.DOTALL)
    html = re.sub(r'<title>.*?</title>', f'<title>{', '.join(pmc)}</title>', html, count=1, flags=re.DOTALL)

    # with open('output.html', 'w', encoding='utf-8', newline='') as f:
    #     f.write(html)

    return html


def get_counts(counts, matches, cites):
    matches_done = set()
    for s in matches:
        if s[3]:
            continue
        if s[0] in counts:
            counts_0 = counts[s[0]][0] + (1 if s[0] not in matches_done else 0)
            counts_1 = counts[s[0]][1] + (cites[s[1]] if s[1] not in counts[s[0]][2] else 0)
            counts[s[0]][2].add(s[1])
            counts[s[0]] = counts_0, counts_1, counts[s[0]][2]
        else:
            counts[s[0]] = 1, cites[s[1]], {s[1]}
        matches_done.add(s[0])


def write_counts(file_path, counts):
    counts = dict(sorted(counts.items(), key=lambda x: x[1][0], reverse=True))
    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t', lineterminator='\n')
        writer.writerow(['Match', 'Count', 'Cites'])
        for key, value in counts.items():
            writer.writerow([key, value[0], value[1]])


def get_scores(file_path):
    scores = {}
    with open(file_path, 'r', encoding='utf-8') as file:
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
                            citedByCount = soup.find('citedByCount')
                            if citedByCount:
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
    fulltexts = get_file_path('../biotoolspub/fulltexts')
    counts = {}
    for token_label in Tokens:
        counts[token_label] = {}
    i = 0
    for xml in os.listdir(fulltexts):
        with open(os.path.join(fulltexts, xml), 'r', encoding='utf-8') as file:
            soup = BeautifulSoup('<article>' + file.read() + '</article>', 'xml')
            text = soup.get_text(separator=' ')
        tokens = {Tokens.ORIG: get_tokens(text)}
        tokens[Tokens.CASED] = get_tokens_cased(tokens[Tokens.ORIG])
        tokens[Tokens.LOWER] = get_tokens_lower(tokens[Tokens.ORIG])
        matches = get_matches(tokens, biotools)
        for token_label in Tokens:
            get_counts(counts[token_label], matches[token_label], cites)
        i += 1
        print(f'counts {i}', flush=True)
    for token_label in Tokens:
        write_counts(get_file_path(f'counts_{token_label.value}.tsv'), counts[token_label])


def main():
    # match_xml([
    #     'PMC3257301', 'PMC10845142', 'PMC8479664', 'PMC1160178', 'PMC11330317',
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
    # match_xml('PMC11387482')
    # generate_counts()
    pass


if __name__ == '__main__':
    main()
