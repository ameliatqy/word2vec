import torch
from torch.nn import Embedding, Module, Linear


class SkipGramModel(Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 projection_dim: int,
                 training_context_size: int):
        self._input_embeddings = Embedding(vocab_size, embedding_dim)
        self._projection_layer = Linear(embedding_dim, projection_dim)
        self._output_layer = Linear(projection_dim, training_context_size * embedding_dim)

    def forward(self, input_word):
        input_embeddings = self._input_embeddings(input_word)
        projected_input_embeddings = self._projection_layer(input_embeddings)
        output_embeddings = self._output_layer(projected_input_embeddings)
        return output_embeddings

