# rag_engine.py
class RAGEngine:
    def __init__(self, vector_db, embedding, router, reflection, reranker, chat_client, base_sys_prompt):
        self.vector_db = vector_db
        self.embedding = embedding
        self.router = router
        self.reflection = reflection
        self.reranker = reranker
        self.chat_client = chat_client
        self.base_sys_prompt = base_sys_prompt

    def chat(self, messages, query):
        original_prompt = self.base_sys_prompt

        route_score, best_route = self.router.guide(query)
        context_product = ""

        if best_route == "products":
            rewritten_query = self.reflection.rewrite(messages, query)
            query_embedding = self.embedding.encode(rewritten_query)
            if hasattr(query_embedding, "tolist"):
                query_embedding = query_embedding.tolist()

            results = self.vector_db.query("products_clean", query_embedding, limit=20)
            if results:
                passages = [r["information"] for r in results]
                _, ranked = self.reranker(rewritten_query, passages)
                context_product = "\n\n".join(ranked[:5])

            messages[0]["content"] = original_prompt + "\n\n" + context_product
            messages.append({"role": "user", "content": rewritten_query})
        else:
            messages[0]["content"] = original_prompt
            messages.append({"role": "user", "content": query})

        response = self.chat_client.chat.completions.create(
            model="openai-gpt-oss-120b",
            messages=messages
        )

        answer = response.choices[0].message.content
        messages.append({"role": "assistant", "content": answer})

        return answer, best_route
