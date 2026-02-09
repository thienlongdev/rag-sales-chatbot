import numpy as np

# routes = [
#     Route(name="products", samples=productsSample),
#     Route(name='chitchat', samples=chichatSample)
# ]

# mbedding = Embeddings(model_name='all-MiniLM-L6-v2', type='sentence_transformers')
# router = SemanticRouter(embedding, routes)

class SemanticRouter:
    def __init__(self, embedding, routes):
        self.embedding = embedding
        self.routes = routes
        self.routesEmbedding = {}
        for route in self.routes:
            self.routesEmbedding[route.name] = self.embedding.encode(route.samples)
            # {"products": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]}
    
    def get_routes(self):
        return self.routes
    def guide(self, query):
        queryEmbedding = self.embedding.encode([query])
        queryEmbedding = queryEmbedding / np.linalg.norm(queryEmbedding) # a / ||a||
        scores = [] # [0.4, 'products'], [0.3, 'chitchat']
        for route in self.routes:
            routeEmbedding = self.routesEmbedding[route.name]
            routeEmbedding = routeEmbedding / np.linalg.norm(routeEmbedding) # b / ||b||
            score = np.mean(np.dot(queryEmbedding, routeEmbedding.T).flatten())
            scores.append((score, route.name))
        scores.sort(reverse=True)
        return scores[0]