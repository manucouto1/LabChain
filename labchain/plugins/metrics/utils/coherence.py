import string
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from transformers import BertTokenizer

from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from tqdm.auto import tqdm

tqdm.pandas()


class Coherence:
    def __init__(self, f_vocab=None, topk=10, processes=1, measure="c_npmi"):
        super().__init__()

        if f_vocab is None:
            tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", do_lower_case=True
            )

            self.f_vocab = list(
                token
                for token in tokenizer.get_vocab().keys()
                if token not in set(stopwords.words("english"))
                and token not in string.punctuation
                and not token.startswith("[")
                and not token.startswith("##")
                and pos_tag([token], tagset="universal")[0][-1]
                in ["NOUN", "VERB", "ADJ"]
            )
        else:
            self.f_vocab = f_vocab

        self.topk = topk
        self.processes = processes
        self.measure = measure

    def evaluate(self, df, predicted):
        _, d_y, _ = predicted

        topics = list(d_y.values())
        # Creo que aquí está el origen de todos los males
        print(topics)
        texts = df.text.progress_apply(
            lambda text: [
                word for word in simple_preprocess(text) if word in self.f_vocab
            ]
        )

        texts = texts.drop(texts[texts.apply(len) == 0].index).values.tolist()

        dictionary = Dictionary(texts)

        if self.topk > len(topics[0]):
            raise Exception("Words in topics are less than topk")
        else:
            npmi = CoherenceModel(
                topics=topics,
                texts=texts,
                dictionary=dictionary,
                coherence=self.measure,
                processes=self.processes,
                topn=self.topk,
            )

            return npmi.get_coherence()
