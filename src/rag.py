import openai
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
import os

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')


# PDF 문서에서 텍스트를 추출하고 인덱싱
class RAGChatbot:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.index = None
        self.build_index()

    def build_index(self):
        # 각 문서에서 텍스트 부분만 추출
        text_data = [doc['text'] if isinstance(doc, dict) else doc for doc in self.documents]
        
        # 문서 텍스트 벡터화
        vectors = self.vectorizer.fit_transform(text_data)
        faiss_index = faiss.IndexFlatL2(vectors.shape[1])  # L2 거리 계산
        faiss_index.add(np.array(vectors.toarray(), dtype=np.float32))
        self.index = faiss_index


    def retrieve_relevant_docs(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query]).toarray().astype(np.float32)
        distances, indices = self.index.search(query_vec, top_k)
        return [self.documents[i] for i in indices[0]]

    def generate_answer(self, query):
        relevant_docs = self.retrieve_relevant_docs(query)
        context = "\n".join(doc['text'] if isinstance(doc, dict) else doc for doc in relevant_docs)
        prompt = f"문서에서 얻은 정보:\n{context[:20000]}\n\n질문: {query}\n답변:"

        # 'openai.ChatCompletion.create' 사용
        response = openai.ChatCompletion.create(  # 'completions' -> 'ChatCompletion'
            model="gpt-4o",  # 또는 "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response['choices'][0]['message']['content'].strip()

# document_index.json 파일을 불러와서 documents로 사용
def load_documents(index_file):
    with open(index_file, "r", encoding="utf-8") as f:
        return json.load(f)

