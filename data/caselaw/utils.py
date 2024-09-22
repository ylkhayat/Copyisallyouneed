import re
import nltk
from eyecite import get_citations, clean_text

def clean_line_train(line):
    cleaned_text = clean_text(line, ['html', 'all_whitespace'])
    citations = get_citations(cleaned_text)
    for citation in citations:
        citation_full = citation.corrected_citation_full()
        # print(citation.groups.values())
        if citation_full in cleaned_text:
            cleaned_text = cleaned_text.replace(citation_full, '<citation>')
        else:
            # print(f"[!] {citation_full} not found in {cleaned_text}")
            # break
            cleaned_text = cleaned_text.replace(citation.matched_text(), '<citation>')
    cleaned_text = re.sub(r'(,*\s*)*<citation>', ' <citation>', cleaned_text)
    cleaned_text = re.sub(r'<citation>(\s*,*\s*)*', '<citation>', cleaned_text)
    cleaned_text = re.sub(r'<citation>(\d*-*\d*\s*)*', '<citation>', cleaned_text)
    cleaned_text = re.sub(r'<citation>(\(\w*\)\s*)*', '<citation>', cleaned_text)
    cleaned_text = re.sub(r'<citation>(at (\d+-*\d*)*)*', '<citation>', cleaned_text)
    cleaned_text = re.sub(r'<citation>', '<citation>', cleaned_text)
    cleaned_text = cleaned_text.replace('<citation>', '')
    cleaned_text = cleaned_text.replace(' . ', '. ')
    cleaned_text = cleaned_text.replace(' , ', ', ')
    cleaned_text = cleaned_text.replace(' ; ', '; ')
    cleaned_text = cleaned_text.replace(' : ', ': ')
    cleaned_text = cleaned_text.replace(' ! ', '! ')
    cleaned_text = cleaned_text.replace(' ? ', '? ')
    cleaned_text = cleaned_text.replace(' ( ', ' (')
    cleaned_text = cleaned_text.replace(' ) ', ') ')
    cleaned_text = cleaned_text.replace('“','"')
    cleaned_text = cleaned_text.replace('”','"')
    cleaned_text = cleaned_text.replace('’',"'")
    cleaned_text = cleaned_text.replace('‘',"'")
    cleaned_text = cleaned_text.replace(' - ', '-')
    cleaned_text = cleaned_text.replace(' -', '-')
    cleaned_text = cleaned_text.replace('- ', '-')

    # cleaned_text = re.sub(r'<citation>\s*at\s*(\w*\s*\d*[,-]*)*', '<citation>', cleaned_text)
    # cleaned_text = re.sub(r'<citation> , \d*\s*(\(\d+\w*\))', '<citation>', cleaned_text)
    # cleaned_text = re.sub(r'<citation>\s*[,(and)]*\s*(\d+\.*\d+\s*)*(\(\d+\)\s*)*', '<citation>', cleaned_text)
    # cleaned_text = re.sub(r'<citation>\s*(\(\d+\w*\))', '<citation>', cleaned_text)
    # cleaned_text = re.sub(r'<citation>(\s*\(\d*\w*\)\s*)*', '<citation>', cleaned_text)
    return cleaned_text

    # print(text)
    # # cleaned_text = re.sub(r'\(see(?:[^()]*|\((?:[^()]*|\([^()]*\))*\))*\)', '', line)

    # cleaned_text = re.sub(r'Id. at \d*\w*-*\d*', '', cleaned_text)
    # cleaned_text = cleaned_text.replace('<citation>', '')
    # cleaned_text = re.sub(r'R\. [Aa]t .*?([\.,\)])', r'\1', cleaned_text)
    # cleaned_text = re.sub(r'(,\s*)+', ', ', cleaned_text)
    # cleaned_text = re.sub(r'\.\s*\.', '.', cleaned_text)
    # cleaned_text = re.sub(r',\s*\.', '.', cleaned_text)
    # cleaned_text = cleaned_text.replace(', . ', '. ')
    # cleaned_text = cleaned_text.replace('. . ', '. ')
    # cleaned_text = cleaned_text.replace('. , ', '. ')
    # cleaned_text = cleaned_text.replace(', , ', ', ')
    # cleaned_text = cleaned_text.replace(' , ', ', ')
    # cleaned_text = re.sub(r'(,\s*)*\d+ U\.S\.C\..*?\d+\w*(\s*\(\d*\w*?\))*', '', cleaned_text) # Remove U.S.C. citations
    # cleaned_text = re.sub(r'(,\s*)*\d+ Vet\.App\..*?\d+\w*(\s*\(\d*\w*?\))*', '', cleaned_text) # Remove Vet.App. citations
    # cleaned_text = re.sub(r'(,\s*)*\d+ Vet\.App\.\s*at\s*\d+(-\d+)?(?:,\s*\d+(-\d+)?)*\s*(\(\d{4}\))*', '', cleaned_text) # Remove Vet.App. citations
    # cleaned_text = re.sub(r'(,\s*)*\d* C\.F\.R\.\s*(\s*\d*[\.,]*(and)*)*([Pp]art\s*\d*,*)*\s*([cC]ode\s*\d*)*\s*(\(\d+\))*', '', cleaned_text) # Remove C.F.R. citations
    # cleaned_text = re.sub(r'(,\s*)*\d* C\.F\.R\.\s*(\d*[\.,]*)*.*?(\(\d+\))*', '', cleaned_text) # Remove C.F.R. citations
    # cleaned_text = re.sub(r'(,\s*)*\d+ U\.S\. (\(.*?\))*\s*(at\s*)*(-*\d*,*\s*)*', '', cleaned_text) # Remove U.S. citations
    # cleaned_text = re.sub(r'(,\s*)*\d+ S\.Ct\.\s*(at\s*)*(\d*-*)*', '', cleaned_text) # Remove S.Ct. citations
    # cleaned_text = re.sub(r'(,\s*)*\d+ F\.3d\s*(at\s*)*(\d*-*)*', '', cleaned_text) # Remove F.3d citations
    # cleaned_text = re.sub(r'(,\s*)*\d*\w* F\.2d\s*(at\s*)*(\s*\d*-*\d*[,\[\]]*)*', '', cleaned_text) # Remove F.2d citations
    # return cleaned_text


def clean_line_test(line):
    words = nltk.word_tokenize(line)
    text = ' '.join(words)
    cleaned_text = text.replace('\n', ' ')
    cleaned_text = cleaned_text.replace(' , ', ',')
    cleaned_text = cleaned_text.replace(' .', '.')
    cleaned_text = cleaned_text.replace(' !', '!')
    cleaned_text = cleaned_text.replace(' ?', '?')
    cleaned_text = cleaned_text.replace(' : ', ': ')
    cleaned_text = re.sub(r'(,\s*)+', ', ', cleaned_text)
    return cleaned_text
    
    