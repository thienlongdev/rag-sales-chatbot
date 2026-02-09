from typing import List, Dict

class Reflection:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def rewrite(self, messages: List[Dict], current_query: str) -> str:
        """
        Viết lại current_query thành câu hỏi độc lập từ context.

        :param messages: Lịch sử chat (dạng OpenAI chat messages)
        :param current_query: Câu hỏi hiện tại từ người dùng
        :return: Câu hỏi đã viết lại
        """
        chat_history = [msg for msg in messages if msg['role'] in ('user', 'assistant')][-10:]

        history_text = ""
        for msg in chat_history:
            role = "Khách" if msg['role'] == 'user' else "Bot"
            history_text += f"{role}: {msg['content']}\n"
        history_text += f"Khách: {current_query}\n"
        system_prompt_content = f"""Bạn là một chuyên gia về ngôn ngữ và ngữ cảnh cho hệ thống tìm kiếm AI. Nhiệm vụ của bạn là viết lại câu hỏi mới nhất của người dùng thành một "câu hỏi độc lập" (standalone question).

        Dữ liệu đầu vào bao gồm:
        1. Lịch sử trò chuyện (Chat History): Chứa ngữ cảnh của cuộc hội thoại.
        2. Câu hỏi mới nhất (Latest Question): Câu hỏi người dùng vừa nhập, có thể thiếu chủ ngữ hoặc phụ thuộc vào lịch sử trước đó.

        Quy tắc bắt buộc:
        - Dựa vào Lịch sử trò chuyện để làm rõ các đại từ (nó, cái đó, anh ấy...), các tham chiếu ngầm hoặc ngữ cảnh bị thiếu trong Câu hỏi mới nhất.
        - Viết lại thành một câu hỏi hoàn chỉnh, đầy đủ ý nghĩa mà không cần nhìn vào lịch sử trò chuyện vẫn hiểu được.
        - Giữ nguyên ý định ban đầu của người dùng.
        - KHÔNG ĐƯỢC trả lời câu hỏi. Chỉ viết lại câu hỏi thôi.
        - Nếu câu hỏi mới nhất đã rõ ràng và không liên quan đến lịch sử, hãy giữ nguyên nó.

        Ví dụ minh họa:
        Lịch sử: 
        - User: LangChain là gì?
        - AI: LangChain là một framework để phát triển ứng dụng LLM...
        Câu hỏi mới: Nó hoạt động như thế nào?
        -> Câu hỏi viết lại: LangChain hoạt động như thế nào?

        Lịch sử:
        - User: So sánh ChromaDB và Pinecone.
        - AI: ChromaDB là mã nguồn mở, trong khi Pinecone là dịch vụ quản lý...
        Câu hỏi mới: Cái nào rẻ hơn?
        -> Câu hỏi viết lại: Giữa ChromaDB và Pinecone, cái nào có chi phí rẻ hơn?

        Lịch sử:
        - User: Tôi muốn học Python.
        - AI: Tuyệt vời, bạn có thể bắt đầu với cú pháp cơ bản...
        Câu hỏi mới: Thời tiết Hà Nội hôm nay thế nào?
        -> Câu hỏi viết lại: Thời tiết Hà Nội hôm nay thế nào?

        ---
        BẮT ĐẦU TÁC VỤ:

        Lịch sử trò chuyện:
        {history_text}

        Câu hỏi mới nhất:
        {current_query}

        Câu hỏi viết lại:"""

        
        prompt = [
            {
                "role": "system",
                "content": system_prompt_content
            },
            {
                "role": "user",
                "content": history_text
            }
        ]



        response = self.llm_client.chat.completions.create(
            model='openai-gpt-oss-120b',
            messages=prompt
        )

        rewritten = response.choices[0].message.content.strip()
        print(f"Reflection: \"{rewritten}\"")
        return rewritten