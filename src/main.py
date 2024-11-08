from rag import RAGChatbot, load_documents
from pdf_to_text import extract_text_from_pdf

if __name__ == "__main__":
    # PDF 문서를 텍스트로 변환
    # pdf_files = ['data/documents/컴퓨터프로그래밍01-240902.pdf','data/documents/컴퓨터프로그래밍02-240909.pdf', 'data/documents/컴퓨터프로그래밍03-1-240923.pdf', 'data/documents/컴퓨터프로그래밍03-2-240923.pdf']
    # documents = [extract_text_from_pdf(pdf) for pdf in pdf_files]
    documents = load_documents("data/index/document_index.json")

    # RAG Chatbot 초기화
    chatbot = RAGChatbot(documents)

    # 사용자 입력 받기
    query = input("질문을 입력하세요: ")
    
    # 답변 생성
    answer = chatbot.generate_answer(query)
    print("답변:", answer)
