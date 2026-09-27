
class Reranker:

    def __init__(self, reraank_model):
        self.reranker_model  = reraank_model
        self.threshold = 0.05
    
    def rerank(self, query, results, top_k=5):
        print(f"[DEBUG] Reranking {len(results)} candidates")
        
        pairs = [
            (query, result["metadata"]["text"])
            for result in results
        ]

        scores = self.reranker_model.predict(pairs)

        ranked = sorted(
                   zip(scores, results),
                   key=lambda x: x[0],
                   reverse=True
                )
        # Apply threshold
        filtered = []
    
        for score, result in ranked:
            print(
                f"[DEBUG] page={result['metadata']['page']} "
            f"reranker_score={score:.4f}"
            )
    
            if score >= self.threshold:
                result["reranker_score"] = float(score)
                filtered.append(result)
    
        # Keep only top_k after threshold filtering
        final_results = filtered[:top_k]
    
        print(
            f"[DEBUG] {len(final_results)} results passed "
        f"threshold={self.threshold}"
        )
        print("\n[DEBUG] Retrieved chunk contents:")

        for i, result in enumerate(final_results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Page: {result['metadata']['page']}")
            print(f"Score: {result.get('reranker_score')}")
            print(f"Text:\n{result['metadata']['text']}")

        return final_results