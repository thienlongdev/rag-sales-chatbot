from openai import OpenAI
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

class Embeddings:
    def __init__(self, model_name, type, client = None):
        self.model_name = model_name
        self.type = type
        
        # --- CẤU HÌNH OPENAI / MEG ALLM ---
        if self.type == 'openai':
            if client is None:
                api_key = os.getenv("MEGALLM_API_KEY") 
                if not api_key:
                    raise ValueError("Không tìm thấy MEGALLM_API_KEY trong file .env")
                
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://ai.megallm.io/v1"
                )
            else:
                self.client = client
                
        # --- CẤU HÌNH GOOGLE ---
        elif self.type == 'google':
            pass
            
        # --- CẤU HÌNH LOCAL (Sentence Transformers) ---
        elif self.type == 'sentence_transformers':
            # Tải model về cache ngay khi khởi tạo class
            print(f"Đang tải model {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
    
    def encode(self, doc):
        """
        Hàm encode luôn trả về Python List (List[float])
        để tương thích trực tiếp với MongoDB.
        """
        # 1. Xử lý OpenAI
        if self.type == 'openai':
            if not self.client:
                 raise ValueError("Client OpenAI chưa được khởi tạo!")

            return self.client.embeddings.create(
                input=doc,
                model=self.model_name
            ).data[0].embedding
            """
            {
                "object": "list",
                "data": [
                    {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [
                        0.0123,
                        -0.0456,
                        0.0789,
                        ...
                        384 số float
                    ]
                    }
                ],
                "model": "text-embedding-3-small",
                "usage": {
                    "prompt_tokens": 7,
                    "total_tokens": 7
                }
            }

            """
            
        # 2. Xử lý Google
        elif self.type == 'google':
            pass
            
        # 3. Xử lý Local (Quan trọng: Convert sang list ở đây)
        elif self.type == 'sentence_transformers':
            vector = self.model.encode(doc)
            
            # Tự động chuyển đổi NumPy array sang List chuẩn của Python
            if hasattr(vector, 'tolist'):
                return vector.tolist()
            
            return vector