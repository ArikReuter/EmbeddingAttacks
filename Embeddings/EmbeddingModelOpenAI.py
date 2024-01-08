from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm
import numpy as np
import os
import tiktoken

from Embeddings.EmbeddingModel import EmbeddingModel


class EmbeddingModelOpenAI(EmbeddingModel):
    """
    Embedding model following the OpenAI API.
    """

    def __init__(self,
                 deployment_name: str = "text-embedding-ada-002",
                 model_name: str = "text-embedding-ada-002",
                 context_len: int = 8191,
                 openai_api_key: str = os.environ['OpenAI_Key_DataSecurity'],
                 max_retries: int = 10):

        """

        Args:
            deployment_name (str): name of the deployment to use
            model_name (str): name of the model to use
            context_len (int): maximum number of tokens to use
            openai_api_key (str): API key to use the OpenAI API
            max_retries (int): maximum number of retries to use when calling the OpenAI API

        """

        self.deployment_name = deployment_name
        self.model_name = model_name
        self.max_retries = max_retries
        self.context_len = context_len

        self.embeddings_mod = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            max_retries=max_retries
        )

    def embed_document(self, document: str) -> list[float]:
        """
        Embeds a query
        Args:
            query (str): query to embed

        """
        return self.embeddings_mod.embed_query(document)

    def embed_document_list(self, documents: list[str]) -> list[list[float]]:
        """
        Embeds a list of documents

        Args:
            documents (list[str]): documents to embed

        """
        documents = [str(document) for document in documents]

        return self.embeddings_mod.embed_documents(documents)

    @staticmethod
    def num_tokens_from_string(string: str, encoding) -> int:
        """
        Returns the number of tokens in a text string.

        Args:
            string (str): Text string to compute the number of tokens.
            encoding: A function to encode the string into tokens.

        Returns:
            int: Number of tokens in the text string.
        """
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def compute_number_of_tokens(self, corpus: list[str]) -> int:
        """
        Computes the total number of tokens needed to embed the corpus.

        Args:
            corpus (list[str]): List of strings to embed, where each element in the list is a document.

        Returns:
            int: Total number of tokens needed to embed the corpus.
        """

        if self.tokenizer_str is None:
            tokenizer = tiktoken.encoding_for_model(self.embedding_model)

        else:
            tokenizer = tiktoken.get_encoding(self.tokenizer_str)

        num_tokens = 0
        for document in tqdm(corpus):
            num_tokens += self.num_tokens_from_string(document, tokenizer)

        return num_tokens

    def split_doc(self, text):

        """
        Splits a single document that is longer than the maximum number of tokens into a list of smaller documents.

        Args:

            self: The instance of the class.
            text (str): The string to be split.

        Returns:
            List[str]: A list of strings to embed, where each element in the list is a list of chunks comprising the document.
        """

        split_text = []
        split_text.append(text[:self.context_len])

        for i in range(1, len(text) // self.context_len):
            split_text.append(text[i * self.context_len:(i + 1) * self.context_len])

        split_text.append(text[(len(text) // self.context_len) * self.context_len:])

        return split_text

    def split_long_docs(self, text: list[str]) -> list[list[str]]:

        """
        Splits all documents that are longer than the maximum number of tokens into a list of smaller documents.
        Args:

            self: The instance of the class.
            text (list[str]): List of strings to embed, where each element in the list is a document.

        Returns:

            List[list[str]]: A list of lists of strings to embed, where each element in the outer list is a list of chunks comprising the document.
        
        """

        split_text = []

        for document in tqdm(text):

            encoding = tiktoken.encoding_for_model(self.model_name)
            if self.num_tokens_from_string(document, encoding=encoding) > self.context_len:
                split_text.append(self.split_doc(document))

            else:
                split_text.append([document])

        return split_text

    def get_embeddings_doc_split(self, corpus: list[list[str]], n_tries=3) -> list[np.array]:

        """
        Computes the embeddings of a corpus for split documents.

        Args:

            self: The instance of the class.
            corpus (list[list[str]]): List of strings to embed, where each element is a document represented by a list of its chunks.

        Returns:
            List[np.array]: A list of embeddings, where each element in the list is the embedding of a document.
        """

        api_res_list = []

        for i in tqdm(list(range(len(corpus)))):
            chunk_lis = corpus[i]
            api_res_doc = []

            for chunk_n, chunk in enumerate(chunk_lis):
                try:
                    api_res_doc.append(
                        {"api_res": self.embed_document(chunk),
                         "error": None}
                    )
                    break

                except Exception as e:

                    print(f"Error {e} occured for chunk {chunk_n} of document {i}")
                    print(chunk)
                    api_res_doc.append(
                        {"api_res": None,
                         "error": e})

            # average the embeddings of the chunks
            emb_lis = []

            for api_res in api_res_doc:
                if api_res["api_res"] is not None:
                    emb_lis.append(api_res["api_res"])

            text = " ".join(chunk_lis)
            embedding = np.mean(emb_lis, axis=0)

            api_res_list.append(
                {"embedding": embedding,
                 "text": text,
                 "errors": [api_res["error"] for api_res in api_res_doc]}
            )

        return api_res_list

    def convert_api_res_list(self, api_res_list: list[dict]) -> dict:

        """
        Converts the api_res list into a dictionary containing the embeddings as a matrix and the corpus as a list of strings.

        Args:

            self: The instance of the class.

            api_res_list (list[dict]): List of dictionaries, where each dictionary contains the embedding of the document, the text of the document, and a list of errors that occurred during the embedding process.

        Returns:

            dict: A dictionary containing the embeddings as a matrix and the corpus as a list of strings.
        """

        embeddings = np.array([api_res["embedding"] for api_res in api_res_list])
        corpus = [api_res["text"] for api_res in api_res_list]
        errors = [api_res["errors"] for api_res in api_res_list]

        return {"embeddings": embeddings, "corpus": corpus, "errors": errors}

    def embed_document_list_splitting(self, document_list: list[str]) -> dict:

        """
        Computes the embeddings of a corpus.
        Args:

            self: The instance of the class.
            corpus (list[str]): List of strings to embed, where each element in the list is a document.

        Returns:
            dict: A dictionary containing the embeddings as a matrix and the corpus as a list of strings.

        """

        corpus_split = self.split_long_docs(document_list)
        corpus_emb = self.get_embeddings_doc_split(corpus_split)
        self.corpus_emb = corpus_emb
        res = self.convert_api_res_list(corpus_emb)
        return res
