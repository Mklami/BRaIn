import re
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

# Add src directory to Python path
script_dir = Path(__file__).parent.parent.parent.absolute()
src_dir = script_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from Utils.NLP.TextPreprocessor import TextPreprocessor

class TextRank:

    IDF = {}
    def __init__(self):
        self.graph = nx.Graph()
        self.textProcessor = TextPreprocessor(remove_SE_stop_words=True)



    def get_keywords_CodeRank(self, query, documents, no_of_keywords=10, window_size=20):
        self.graph = nx.Graph()
        query_tokens = self.preprocess_text(query)

        document_tokens_by_document = []
        bias_dict = {}

        # Preprocess documents and build initial token data
        for document in documents:
            # javaParser = JavaSourceParser(document)
            # class_token, method_token, field_token = javaParser.parse_class_method_field_name(document)
            temp_tokens = self.preprocess_text(document)

            # Use a set for efficient membership checks
            # all_tokens = set(temp_tokens) | set(class_token) | set(method_token) | set(field_token)
            # all_tokens = set(temp_tokens)
            document_tokens_by_document.append(temp_tokens)

        # Calculate bias dict for query tokens
        for token in query_tokens:
            if token not in bias_dict:
                # Guard against empty queries after preprocessing
                if len(query_tokens) == 0:
                    continue
                tf_query = query_tokens.count(token) / len(query_tokens)
                idf = self.IDF.get(token, 0)
                bias_dict[token] = tf_query * idf

        # If we couldn't build any personalization/bias (e.g., empty query), fall back.
        if not bias_dict:
            # Return most frequent query tokens (deduped) as a lightweight fallback.
            seen = set()
            fallback = []
            for t in query_tokens:
                if t not in seen:
                    seen.add(t)
                    fallback.append(t)
                if len(fallback) >= no_of_keywords:
                    break
            return fallback

        document_token_indices = {}
        for doc_id, document in enumerate(document_tokens_by_document):
            token_positions = defaultdict(list)
            for i, token in enumerate(document):
                token_positions[token].append(i)
            document_token_indices[doc_id] = token_positions

        self.graph = nx.Graph()

        # Build graph edges
        for token in query_tokens:
            if token in bias_dict:
                for doc_id, token_positions in document_token_indices.items():
                    if token in token_positions:
                        indices = token_positions[token]
                        for index in indices:
                            start = max(0, index - window_size)
                            end = min(len(document_tokens_by_document[doc_id]), index + window_size)
                            document = document_tokens_by_document[doc_id]

                            for i in range(start, end):
                                neighbor_token = document[i]
                                if neighbor_token in bias_dict:
                                    self.graph.add_edge(token, neighbor_token, weight=self.graph[token][neighbor_token][
                                                                                          'weight'] + 1 if self.graph.has_edge(
                                        token, neighbor_token) else 1)

                                    # Handle camel case tokens
                                    camel_case_tokens = self.split_camel_case(neighbor_token)
                                    for camel_token in camel_case_tokens:
                                        if camel_token in bias_dict:
                                            self.graph.add_edge(token, camel_token,
                                                                weight=self.graph[token][camel_token][
                                                                           'weight'] + 1 if self.graph.has_edge(token,
                                                                                                                camel_token) else 1)

        # If graph ended up empty, PageRank will fail. Fall back gracefully.
        if self.graph.number_of_nodes() == 0 or self.graph.number_of_edges() == 0:
            seen = set()
            fallback = []
            for t in query_tokens:
                if t not in seen and t in bias_dict:
                    seen.add(t)
                    fallback.append(t)
                if len(fallback) >= no_of_keywords:
                    break
            return fallback

        # Compute PageRank scores
        try:
            scores = nx.pagerank(
                self.graph,
                alpha=0.85,
                weight="weight",
                max_iter=1000,
                personalization=bias_dict,
            )
        except ZeroDivisionError:
            # Known failure mode when normalization constants are zero in scipy backend
            # (e.g., due to degenerate personalization/graph).
            seen = set()
            fallback = []
            for t in query_tokens:
                if t not in seen and t in bias_dict:
                    seen.add(t)
                    fallback.append(t)
                if len(fallback) >= no_of_keywords:
                    break
            return fallback

        # Return top keywords
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [token for token, score in sorted_scores[:no_of_keywords]]

    # Backwards-compatible alias used by some scripts.
    def get_keywords_CodeRank_3(self, query, documents, no_of_keywords=10, window_size=20):
        return self.get_keywords_CodeRank(
            query=query,
            documents=documents,
            no_of_keywords=no_of_keywords,
            window_size=window_size,
        )



    def preprocess_text(self, text):
        return self.textProcessor.preprocess(text)

    def split_camel_case(self, param):
        """
                Splits a camelCase or PascalCase identifier into individual words.
                For example, 'isTrue' becomes ['is', 'True'].
                """
        return re.sub('([a-z])([A-Z])', r'\1 \2', param).split()









