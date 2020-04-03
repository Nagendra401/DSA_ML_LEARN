
import pandas as pd
from nltk import word_tokenize
from nltk import pos_tag    #  parts of speech tagging.
from nltk import ne_chunk   # Named Entity Recognition.
from nltk import Tree       # To handle tree object returned by ne_chunk.
from nltk.sem.relextract import NE_CLASSES  # To filter classes


path = 'C:\\Users\\Jayendra Vadrevu\\Google Drive\\Darius\\1. DSA\\Course Material\\6. Text Analytics\\R and Python\Python\\Infosys.xlsx' # set reviews file path.

raw_reviews = pd.read_excel(path,sheet_name=0,names=['reviews'])


Text = str(list(raw_reviews.reviews))  # Convert reviews as Single text.
Text
postagged = pos_tag(word_tokenize(Text))
posdf = pd.DataFrame(postagged, columns=['Word', 'Tag'])

entities_tagged = ne_chunk(pos_tag(word_tokenize(Text)))    # ne_chunk needs words to be pos-tagged.
print(entities_tagged)


# Visualizing the identified classes.
"""NE_CLASSES = {
    'ieer': ['LOCATION', 'ORGANIZATION', 'PERSON', 'DURATION',
            'DATE', 'CARDINAL', 'PERCENT', 'MONEY', 'MEASURE'],
    'conll2002': ['LOC', 'PER', 'ORG'],
    'ace': ['LOCATION', 'ORGANIZATION', 'PERSON', 'DURATION',
            'DATE', 'CARDINAL', 'PERCENT', 'MONEY', 'MEASURE', 'FACILITY', 'GPE'-Geo Political Entity],
    }"""
ace_tags = NE_CLASSES['ace']        # Pre-built  defined classes.
# ieer_tags = NE_CLASSES['ieer']      # Use any of them.

for node in entities_tagged:
    if type(node) == Tree and node.label() in ace_tags:  # or ieer_tags
        words, tags = zip(*node.leaves())
        print( ' '.join(words)+'\t'+ node.label())

# End of script.
