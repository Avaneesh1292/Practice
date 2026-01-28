from langfuse import observe

@observe(name="rag_evaluation")
def run_evaluation(eval_data, retriever, rag_fn):
    results = []

    for item in eval_data:
        docs, _ = rag_fn(item["question"], retriever)

        recall = int(any(item["expected"] in d.page_content.lower() for d in docs[:5]))
        mrr = 0
        for i, d in enumerate(docs):
            if item["expected"] in d.page_content.lower():
                mrr = 1 / (i + 1)
                break

        # return metrics instead of logging explicitly
        results.append({
            "recall@5": recall,
            "mrr": mrr
        })

    return results