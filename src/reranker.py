
class Reranker:

    def __init__(self, reraank_model):
        self.reranker_model  = reraank_model
    
    def rerank(self, query, results, top_k=5):
        
        candidate_k = max(top_k * 3, 10)
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

        return [result for score, result in ranked[:top_k]]