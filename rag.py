from embeddings import Embeddings
from vector_db import VectorDatabase
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI 
from semantic_router.route import Route
from semantic_router.router import SemanticRouter
from semantic_router.samples import productsSample, chitchatSample
from reflection import Reflection
from reranker import Reranker
import logging
import warnings
warnings.filterwarnings("ignore") 
logging.getLogger("transformers").setLevel(logging.ERROR)
load_dotenv()

def build_combine_row(row):
    combine = f"Tên sản phẩm: {row['title']}\n"
    combine += f"Mô tả: {row['product_specs']}\n"
    combine += f"Giá: {row['current_price']}\n"
    combine += f"Ưu đãi: {row['product_promotion']}\n"
    combine += f"Màu sắc: {row['color_options']}\n"
    return combine

def main():
    df = pd.read_csv(r"C:\Users\ADMIN\Desktop\AI\Databases\hoanghamobile.csv")
    df = df.fillna('') 
    df['information'] = df.apply(build_combine_row, axis = 1)
    
    vector_db = VectorDatabase(db_type = 'mongodb')
    # vector_db.client.delete_collection("products_clean")
    embedding = Embeddings(model_name='all-MiniLM-L6-v2', type='sentence_transformers')
    routes = [
        Route(name="products", samples=productsSample),
        Route(name="chitchat", samples=chitchatSample)
    ]
    router = SemanticRouter(embedding, routes)
    inserted_count = 0
    if vector_db.count_documents(collection_name='products_clean') == 0:
        for index, row in df.iterrows():
            vector_data = embedding.encode(row['information'])
            if hasattr(vector_data, 'tolist'):
                vector_data = vector_data.tolist()
            
            document = {
                "title": row['title'],
                "product_specs": row['product_specs'],
                "current_price": row['current_price'],
                "product_promotion": row['product_promotion'],
                "color_options": row['color_options'],
                "information": row['information'],
                "embedding": vector_data 
            }
            vector_db.insert_document(collection_name='products_clean', document=document)
            inserted_count += 1
            print(f"Inserted {inserted_count} documents", end = '\r')
            
        if inserted_count == 0:
            print("\nAll documents are already inserted.")
        else:
            print(f"\nInserted {inserted_count} new documents")
        print("Đã tải dữ liệu xong\n")
    else:
        print("Dữ liệu đã tồn tại trong MongoDB, bỏ qua bước insert")

    
    chat_client = OpenAI(
        api_key=os.getenv("MEGALLM_API_KEY"),
        base_url="https://ai.megallm.io/v1"
    )
    base_sys_prompt = """Bạn là một nhân viên tư vấn bán hàng chuyên nghiệp tại cửa hàng Thiên Long Phone.
Chỉ sử dụng thông tin có trong dữ liệu, không tự tạo ra thông tin bạn không được cung cấp.
Nếu không tìm thấy câu trả lời, hãy lịch sự trả lời rằng hiện tại bạn chưa có đủ thông tin.
Nếu trong dữ liệu có giá, PHẢI trả lời chính xác giá.
KHÔNG được nói "chưa có thông tin" nếu dữ liệu đã cung cấp."""

    messages = [{"role": "system", "content": base_sys_prompt}]
    reflection = Reflection(chat_client)
    reranker = Reranker()
    while(True):
        query = input("Ban: ")
        if query.lower().strip() == 'quit':
            print("Cảm ơn bạn đã sử dụng dịch vụ, hẹn gặp lại.")
            break
        
        # RAG

        if 'original_prompt' not in locals():
            original_prompt = messages[0]['content']
            
        route_result = router.guide(query)
        best_route = route_result[1]
        print(f"Đang định tuyến đến route: {best_route} (Score: {route_result[0]:.4f})")
        
        context_product = "" # Biến chứa thông tin sản phẩm tìm được
        if best_route == 'products':
            
            rewritten_query = reflection.rewrite(messages, query) 
            query_embedding = embedding.encode(rewritten_query)
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            
            results = vector_db.query("products_clean", query_embedding, limit=20)
            if results:
                # 2. Tách lấy danh sách text để đưa vào Reranker
                passages = [res['information'] for res in results]
                
                # 3. Gọi Reranker chấm điểm lại (Query đã rewrite + Danh sách sản phẩm)
                ranked_scores, ranked_passages = reranker(rewritten_query, passages)
                
                # 4. Chỉ lấy Top 5 kết quả có điểm cao nhất
                top_k_passages = ranked_passages[:5]
                
                context_product = "\n\nDữ liệu sản phẩm liên quan (Đã Rerank):\n"
                for info in top_k_passages:
                    context_product += f"{info}\n"
            messages[0]["content"] = original_prompt + context_product
            messages.append({"role": "user", "content": rewritten_query})

        else:
            messages[0]["content"] = original_prompt
            messages.append({"role": "user", "content": query})

        # response
        response = chat_client.chat.completions.create(
            model="openai-gpt-oss-120b",
            messages=messages
        )
        reply = response.choices[0].message.content.strip()
        print("Trả lời: ", reply)
        print('-' * 80)
        messages.append({"role": "assistant", "content": reply})


if __name__ == '__main__':
    main()