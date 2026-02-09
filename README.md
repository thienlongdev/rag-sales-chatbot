\# RAG Sales Chatbot



RAG Sales Chatbot là một hệ thống chatbot bán hàng thông minh được xây dựng theo kiến trúc Retrieval-Augmented Generation (RAG), cho phép chatbot:



* Hiểu câu hỏi người dùng theo ngữ nghĩa



* Truy xuất chính xác thông tin sản phẩm từ dữ liệu riêng



* Trả lời tự nhiên, đúng ngữ cảnh và nhất quán như một nhân viên bán hàng



\# Project Structure

```

rag-sales-chatbot/

├── app.py                  # Entry point (chat interface)

├── rag.py                  # High-level RAG pipeline

├── rag\_engine.py           # Core RAG logic

├── semantic\_router/        # Route classification

├── embeddings.py           # Text embedding module

├── vector\_db.py            # Vector database (MongoDB / FAISS)

├── reranker.py             # Reranking retrieved documents

├── reflection.py           # Self-reflection \& answer refinement

├── requirements.txt        # Dependencies

├── .env.example            # Environment variable template

└── README.md

```



\# Tech Stack



* Language: Python 3.10+



* LLM: OpenAI / compatible LLM API



* Embeddings: Sentence-Transformers



* Vector Database: MongoDB



* Routing: Semantic Router



* Framework: Custom RAG implementation



\# Installation

\## Clone repository

```

git clone https://github.com/your-username/rag-sales-chatbot.git

cd rag-sales-chatbot

```

\## Create virtual environment

```

python -m venv venv

source venv/bin/activate   # Linux / Mac

venv\\Scripts\\activate      # Windows

```

\## Install dependencies

```

pip install -r requirements.txt

```

\# Environment Variables

\## Tạo file .env:

```

MONGODB\_URI=your\_mongodb\_uri

MEGALLM\_API\_KEY=your\_api\_key

```

\# Run the Chatbot

```

python app.py

```

\## Ví dụ:

```

User: Tôi muốn mua điện thoại dưới 5 triệu pin trâu

Bot: Mình gợi ý cho bạn Xiaomi Redmi 12...

```



















