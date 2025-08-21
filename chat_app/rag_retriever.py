from .rag_store import RAGStore


class RAGRetriever:
    """Encapsulates retrieval and prompt-building logic using a RAGStore."""

    def __init__(self, store: RAGStore):
        self.store = store

    # ---------- retrieval API -------------------------------------------

    def query(self, query_text, n_results=5, where=None, include=("documents", "metadatas", "distances")):
        """Query the vector store via the associated RAGStore."""
        q_emb = self.store.embedder.embed([query_text])[0]

        kwargs = {
            "query_embeddings": [q_emb],
            "n_results": n_results,
            "include": list(include),
        }
        if where:
            kwargs["where"] = where

        return self.store.collection.query(**kwargs)

    def new_prompt_and_sources(self, prompt: str, n_results: int = 5):
        """Return a prompt augmented with retrieved context and the source metadata."""
        results = self.query(prompt, n_results, include=("documents", "metadatas", "distances"))
        texts_nested = results.get("documents", [[]])
        metas_nested = results.get("metadatas", [[]])
        dists_nested = results.get("distances", [[]])

        contex = '\n'.join(texts_nested[0]) if (texts_nested and texts_nested[0]) else ""
        contexted_prompt = f"From User: {prompt}\nContext to base your answer: {contex}"

        sources = []
        for i in range(len(metas_nested[0]) if metas_nested else 0):
            m = metas_nested[0][i] or {}
            d = (
                dists_nested[0][i]
                if (dists_nested and dists_nested[0] and i < len(dists_nested[0]))
                else None
            )
            sources.append(
                {
                    "source_file": m.get("source_file"),
                    "chunk_index": m.get("chunk_index"),
                    "page": m.get("page", -1),
                    "type": m.get("type", "text"),
                    "distance": d,
                }
            )
        return contexted_prompt, sources
