"""
    Referencia: https://realpython.com/nltk-nlp-python/
    
    - Esto es un ejercicio de Procesamiento de Lenguaje Natural. Es NPL es una rama de la 
    inteligencia artificial que se enfoca en el análisis y comprensión de texto humano y 
    en la interacción con las personas a través del lenguaje natural.
"""


"""
    Tokenizar el texto. Sacamos las palabras clave del texto.
    - La tokenización es el proceso de dividir un texto en unidades más pequeñas, como palabras, frases u oraciones. En este ejemplo usamos
    word_tokenize para hacer la división en palabras.
"""


# Instalar las tools que necesitamos dentro de nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('tagsets')
nltk.download("maxent_ne_chunker")
nltk.download("words")

example_string = """Muad'Dib learned rapidly because his first training was in how to learn. And the first lesson of all was the basic trust that he could learn. It's shocking to find how many people do not believe they can learn, and how many more believe learning to be difficult."""
tokenized_word = word_tokenize(example_string)

# Stop words son las palabras que queremos ignorar
stop_words = set(stopwords.words("english"))

filtered_list = []
for word in tokenized_word:
    if word.casefold() not in stop_words:
        filtered_list.append(word)

filtered_list = [
    word for word in tokenized_word if word.casefold() not in stop_words
]

print("\n\n---- PALABRAS TOKENIZADAS IGNORANDO LAS STOP WORDS ----\n\n")
print(filtered_list)

"""
    Stemming. Consiste en reducir las palabras a su raiz. Por ejemplo "helper" y "helping"
    comparten la raiz "help".
    
    - Understemming happens when two related words should be reduced to the same stem but arent. 
    This is a false negative.
    - Overstemming happens when two unrelated words are reduced to the same stem even though they 
    shouldnt be. This is a false positive.
"""

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_list]

print("\n\n---- STEMMING ----\n\n")
print(stemmed_words)

"""
    Tagging Parts of Speech. Part of speech is a grammatical term that deals with the 
    roles words play when you use them together in sentences.
"""

parts_of_speech = nltk.pos_tag(filtered_list)

print("\n\n---- PARTS OF SPEECH ----\n\n")
print(parts_of_speech)

print(nltk.help.upenn_tagset())

"""
    Lemmatizing. Reduce palabras a su significado raíz, pero aquí, a diferencia de Stemming, obtenemos una palabra completa
    que tiene sentido por si misma en vez de un fragmento de palabra.
"""

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_list]

print("\n\n---- LEMMATIZING ----\n\n")
print("Tokenizing original", filtered_list)
print("Semmatizing ", stemmed_words)
print("Lemmatizing", lemmatized_words)

# Por ejemplo si tenemos la palabra "worst" nos interesa que el lema sea "bad"
worstLemmatizer = lemmatizer.lemmatize("worst")

# Sin embargo podemos ver que obtenemos "worst" porque el lemmatizer ha asumido que "worst" es un sustantivo
print("Lemmatizing the 'worst' word", worstLemmatizer)

# Vamos indicarle al lemmatize que "worst" es un adjetivo. El valor por defecto en pos es "n" de noun, sustantivo
worstLemmatizer = lemmatizer.lemmatize("worst", pos="a")

print("Lemmatizing the 'worst' word as adjective", worstLemmatizer)

"""
    Chunking. Mientras que la tokenizacion nos permite identificar palabras y sentencias, el chunking nos permimte identificar frases.
    Búsqueda de patrones en un texto.
    
    Frase -> Grupo de palabras que no componen un sentido completo. "A planet"
    Sentencia/Oracion -> Grupo de palabras que sí componen un sentido completo.
"""

# FraseNominal: Determinante opcional (?) + cualquier número (*) de adjetivo/numeral/ordinal + sustantivo
grammar_expression = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar_expression)
tree = chunk_parser.parse(parts_of_speech)

print("\n\n---- CHUKING ----\n\n")
# tree.draw()

another_quote = "It's a dangerous business, Frodo, going out your door."
words_in_another_quote = word_tokenize(another_quote)
lotr_pos_tags = nltk.pos_tag(words_in_another_quote)

tree2 = chunk_parser.parse(lotr_pos_tags)
# tree2.draw()

"""
    Chinking. Se usa junto al chunking, pero mientras que el chunking se usa para incluir un patrón, el chinking es para
    excluir un patrón.
"""

# En este caso tenemos una linea por regla.
# La primera regla ({<.*>+})  Se mete la regla entre {} porque son los patrones que queremos incluir en nuestro chunk
# La segunda regla (JJ)  Se mete la regla entre {} porque son los patrones que queremos excluir
grammar_expression = """
    Chunk: {<.*>+}
           }<JJ>{"""
chunk_parser = nltk.RegexpParser(grammar_expression)

tree3 = chunk_parser.parse(lotr_pos_tags)

"""
    En este arbol podemos ver que el adjetivo "dangerous" se excluye (no aparece en ningun chunk).
    El primer chunk, el de la izquierda, tiene todo el texto que aparecia antes de que el adjetivo fuera excluido
    El segundo chunk contiene todo el texto que aparecia despues de que el adjetivo fuera excluido
"""
print("\n\n---- CHUKING & CHINKING ----\n\n")
# tree3.draw()

"""
    Named Entity Recognition (NER)
"""

tree4 = nltk.ne_chunk(lotr_pos_tags)

print("\n\n---- NAMED ENTITY RECOGNITION ----\n\n")
# tree4.draw()


def extract_named_entities(quote, language="english"):
    words = word_tokenize(quote, language=language)
    tags = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tags, binary=True)
    return set(
        " ".join(i[0] for i in t)
        for t in tree
        if hasattr(t, "label") and t.label() == "NE"
    )


quote = """
    Men like Schiaparelli watched the red planet—it is odd, by-the-bye, that
    for countless centuries Mars has been the star of war—but failed to
    interpret the fluctuating appearances of the markings they mapped so well.
    All that time the Martians must have been getting ready.

    During the opposition of 1894 a great light was seen on the illuminated
    part of the disk, first at the Lick Observatory, then by Perrotin of Nice,
    and then by other observers. English readers heard of it first in the
    issue of Nature dated August 2."""

print(extract_named_entities(quote))

spanish_text = "Peter Gene Hernández (Honolulú, Hawái, 8 de octubre de 1985), conocido artísticamente como Bruno Mars, es un cantante, compositor, productor musical y bailarín estadounidense."
print(extract_named_entities(spanish_text, "spanish"))
