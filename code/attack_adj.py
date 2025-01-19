import json
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
import re
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Preprocessing function
def preprocessing(txt):
    """Lowercase the text and remove punctuations and digits."""
    txt = txt.lower()
    txt = re.sub(r'[^\w\s]', '', txt)  # Keep only alphanumeric and spaces
    txt = re.sub(r'\d', '', txt)       # Remove digits
    return txt


def preprocess_revw(revw):
    """Apply preprocessing to a review."""
    revw1 = []
    for word in revw.split():
        revw1.append(preprocessing(word))
    return " ".join(revw1)


def get_synonyms(word, pos_tag):
    """Find synonyms of a word matching the specific part of speech."""
    tag_map = {
        'JJ': wordnet.ADJ, 'JJS': wordnet.ADJ, 'JJR': wordnet.ADJ,
        'NN': wordnet.NOUN, 'NNS': wordnet.NOUN, 'NNP': wordnet.NOUN, 'NNPS': wordnet.NOUN,
        'RB': wordnet.ADV, 'RBR': wordnet.ADV, 'RBS': wordnet.ADV
    }
    wn_tag = tag_map.get(pos_tag, None)
    synonyms = []
    if wn_tag:
        for syn in wordnet.synsets(word, pos=wn_tag):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
    return synonyms


def synonym_attack_dict(dictionary, pos_filter):
    """Create a dictionary for synonym replacement."""
    syn_attack = {}
    for word in dictionary.keys():
        # Iterate through each POS tag in pos_filter
        for pos in pos_filter:
            synonyms = get_synonyms(word, pos)
            for syn in synonyms:
                if syn in dictionary.keys() and syn != word:  # Avoid self-replacement
                    syn_attack[word] = syn
                    break
    return syn_attack


def synonym_attack(review, dictionary, attackType):
    """Perform synonym attack on a review."""
    # Define parts of speech to target
    notations = [['JJ', 'JJS', 'JJR'],  # Adjectives
                 ['NN', 'NNS', 'NNP', 'NNPS'],  # Nouns
                 ['RB', 'RBR', 'RBS']]  # Adverbs

    # Create synonym attack dictionary based on attack type
    pos_filter = notations[attackType]
    syn_attack = synonym_attack_dict(dictionary, pos_filter)

    # Tokenize and tag parts of speech
    words_with_tags = pos_tag(word_tokenize(review))

    # Replace words in the review with their synonyms
    for word, pos in words_with_tags:
        if pos in pos_filter and word in syn_attack:
            # Replace only whole words
            review = re.sub(r'\b' + re.escape(word) + r'\b', syn_attack[word], review)
    return review


def attack_run(review):
    """Main function to perform synonym attack."""
    with open('big_ai_adjective.json') as f:
        dictionary = json.load(f)
    attackType = 0  # 0 for adjectives, 1 for nouns, 2 for adverbs
    preprocessed_review = preprocess_revw(review)
    return synonym_attack(preprocessed_review, dictionary, attackType)


# Example usage
if __name__ == "__main__":
    sample_review = "The movie was absolutely desirable and the actors performed brilliantly!"
    print("Original Review:", sample_review)
    print("Attacked Review:", attack(sample_review))
