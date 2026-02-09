from pymongo import MongoClient
from chromadb import HttpClient
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from supabase import create_client, Client
from dotenv import load_dotenv
import os

# vecttordb = VectorDatabase(dp_type='mongodb')
load_dotenv()

class VectorDatabase:
    def __init__(self, db_type: str):
        self.db_type = db_type
        if self.db_type == 'mongodb':
            self.client = MongoClient(os.getenv('MONGODB_URI'))
        elif self.db_type == 'chromadb':
            self.client = HttpClient(
                host='localhost',
                port=8123
            )
        elif self.db_type == 'qdrant':
            self.client = QdrantClient(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                api_key=os.getenv("QDRANT_API_KEY")  
            )
        elif self.db_type == 'supabase':
            url: str = os.getenv("SUPABASE_URL")
            key: str = os.getenv("SUPABASE_KEY")
            supabase: Client = create_client(url, key)
            self.client = supabase
    def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure collection exists for Qdrant, create if it doesn't
        """
        if self.db_type == "qdrant":
            if not self.client.collection_exists(collection_name=collection_name):
                print(f"[Info] Collection '{collection_name}' not found. Creating it...")

                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=384,
                        distance=qdrant_models.Distance.COSINE
                    )
                )

                print("[Info] Creating index for 'title' field...")
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="title",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD
                )

                return True  

        return False 

    def insert_document(self, collection_name: str, document: dict):
        if self.db_type == 'mongodb':
            db = self.client.get_database('vector_db')
            collection = db[collection_name]
            collection.insert_one(document)
        elif self.db_type == 'chromadb':
            collection = self.client.get_or_create_collection(
                name=collection_name
            )

            metadata = {
                "product_specs": str(document.get("product_specs", "")),
                "current_price": str(document.get("current_price", "")),
                "product_promotion": str(document.get("product_promotion", "")),
                "color_options": str(document.get("color_options", ""))
            }

            collection.add(
                documents=[document["information"]],
                embeddings=[document["embedding"]],
                metadatas=[metadata],
                ids=[document["title"]]
            )
        elif self.db_type == 'qdrant':
            self._ensure_collection_exists(collection_name)
            payload = document.copy()
            if "embedding" in payload:
                del payload["embedding"] 

            self.client.upsert(
                collection_name=collection_name,
                points=[
                    {
                        "id": hash(document["title"]) % (2**63),
                        "vector": document["embedding"],
                        "payload": payload 
                    }
                ]
            )
        elif self.db_type == 'supabase':
            self.client.table(collection_name).insert(document).execute()
        
    def query(self, collection_name: str, query_vector: list, limit: int = 5):
        if self.db_type == 'mongodb':
            db = self.client.get_database('vector_db')
            collection = db[collection_name]
            results = collection.aggregate([
                {
                    "$vectorSearch": {
                        "index": "local_index",
                        "queryVector": query_vector,
                        "path": "embedding",
                        "numCandidates": 100,
                        "limit": limit
                    }
                }
            ])
            return list(results)

        elif self.db_type == 'chromadb':
            collection = self.client.get_or_create_collection(
                name=collection_name
            )
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=limit
            )

            docs = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    meta = results["metadatas"][0][i]
                    
                    docs.append({
                        "title": results["ids"][0][i],
                        "information": results["documents"][0][i],
                        "current_price": meta.get("current_price", "Liên hệ"),
                        "product_specs": meta.get("product_specs", ""),
                        "product_promotion": meta.get("product_promotion", ""),
                        "color_options": meta.get("color_options", "")
                    })

            return docs
        elif self.db_type == 'qdrant':
            if not self.client.collection_exists(collection_name):
                print(f"[Warning] Collection '{collection_name}' doesn't exist for querying")
                return []

            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector, 
                limit=limit
            )
            
            results = response.points 

            formatted_results = []
            for result in results:
                item = result.payload  
                item['score'] = result.score
                formatted_results.append(item)

            return formatted_results
        elif self.db_type == 'supabase':
            try:
                response = self.client.rpc(
                    'match_products_clean',
                    {
                        'query_embedding': query_vector,
                        'match_threshold': 0.0, 
                        'match_count': limit
                    }
                ).execute()
                
                results = response.data 
                formatted_results = []
                for item in results:
                    formatted_results.append({
                        "title": item.get('title'),
                        "information": item.get('information'),
                        "current_price": item.get('current_price'),
                        "product_specs": item.get('product_specs', ''),
                        "product_promotion": item.get('product_promotion', ''),
                        "color_options": item.get('color_options', ''),
                        "score": item.get('similarity')
                    })
                
                return formatted_results
            except Exception as e:
                return []

    def count_documents(self, collection_name: str):
        if self.db_type == 'mongodb':
            db = self.client.get_database('vector_db')
            collection = db[collection_name]
            return collection.count_documents({})
        elif self.db_type == 'chromadb':
            pass
        elif self.db_type == 'qdrant':
            pass
        elif self.db_type == 'supabase':
            pass