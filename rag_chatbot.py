import os
import argparse
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pypdf
from openai import OpenAI

# -------------------------------------------------------
# 상수 정의 및 로컬 임베딩 모델 캐싱
# -------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = APP_DIR / "modeling"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser().resolve()
PAPERS_DIR = APP_DIR / "LLM논문"
DB_PATH = MODEL_DIR / "vector_db.pkl"

_embedder_instance = None

def get_local_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        from sentence_transformers import SentenceTransformer
        print("로컬 임베딩 모델(jhgan/ko-sroberta-multitask) 로딩 중...")
        _embedder_instance = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return _embedder_instance

# -------------------------------------------------------
# 스마트 텍스트 분할 함수
# -------------------------------------------------------
def split_text(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    """텍스트를 문장 경계를 우선시하여 분할합니다."""
    # 공백 정규화
    cleaned_text = " ".join(text.split())
    chunks = []
    start = 0
    text_len = len(cleaned_text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # 문장 마지막 구두점(. ! ?) 근처에서 끊도록 조정
        if end < text_len:
            # 끝 지점 기준 역방향으로 60자 내에서 문장 경계 검색
            found_boundary = False
            for boundary in ['. ', '! ', '? ']:
                idx = cleaned_text.rfind(boundary, end - 60, end)
                if idx != -1:
                    end = idx + 1  # 구두점 포함
                    found_boundary = True
                    break
            # 문장 경계를 찾지 못했다면 단어 경계(공백)에서 분할
            if not found_boundary:
                space_idx = cleaned_text.rfind(' ', end - 20, end)
                if space_idx != -1:
                    end = space_idx
        
        chunk = cleaned_text[start:end].strip()
        if len(chunk) > 15:
            chunks.append(chunk)
            
        start += (chunk_size - overlap)
        if start >= end:
            start = end + 1
            
    return chunks

# -------------------------------------------------------
# RAG 시스템 메인 클래스
# -------------------------------------------------------
class ParkinsonRAG:
    def __init__(self, api_key: str = None):
        """API 키를 초기화합니다. 명시적으로 전달되지 않으면 환경 변수 등에서 가져옵니다."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        self.db = None
        self.load_db()

    def set_api_key(self, api_key: str):
        """런타임에 API 키를 재설정합니다."""
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def is_api_key_valid(self) -> bool:
        """API 키가 정상적으로 설정되었는지 체크합니다."""
        return self.client is not None

    def load_db(self) -> bool:
        """로컬 벡터 DB를 메모리로 불러옵니다."""
        if DB_PATH.exists():
            try:
                with open(DB_PATH, "rb") as f:
                    self.db = pickle.load(f)
                return True
            except Exception as e:
                print(f"벡터 DB 로드 실패: {e}")
                self.db = None
        return False

    def build_db_from_pdfs(self) -> Tuple[int, str]:
        """LLM논문 폴더 안의 PDF들로부터 벡터 DB를 구축합니다."""
        if not PAPERS_DIR.exists():
            return 0, f"논문 폴더가 없습니다: {PAPERS_DIR}"

        pdf_files = list(PAPERS_DIR.glob("*.pdf"))
        if not pdf_files:
            return 0, f"PDF 논문을 찾을 수 없습니다: {PAPERS_DIR}"

        chunks_data = []
        for pdf_path in pdf_files:
            try:
                reader = pypdf.PdfReader(pdf_path)
                filename = pdf_path.name
                
                print(f"파싱 중: {filename} (총 {len(reader.pages)}페이지)")
                for page_idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if not text.strip():
                        continue
                    
                    page_chunks = split_text(text)
                    for chunk in page_chunks:
                        chunks_data.append({
                            "text": chunk,
                            "source": filename,
                            "page": page_idx + 1
                        })
            except Exception as e:
                print(f"PDF 파싱 실패 ({pdf_path.name}): {e}")

        if not chunks_data:
            return 0, "PDF 파일에서 텍스트를 추출하지 못했습니다."

        print(f"총 {len(chunks_data)}개의 텍스트 청크 생성 완료. 로컬 임베딩 벡터 생성 중...")

        try:
            embedder = get_local_embedder()
            batch_texts = [item["text"] for item in chunks_data]
            # sentence-transformers는 자동으로 배치 처리를 지원합니다.
            embeddings = embedder.encode(batch_texts, show_progress_bar=True)
        except Exception as e:
            return 0, f"로컬 임베딩 모델 생성 실패: {e}"

        # 벡터 DB 데이터 생성 및 저장
        db_data = {
            "embeddings": np.array(embeddings, dtype=np.float32),
            "metadata": chunks_data
        }

        # 디렉토리가 없으면 생성
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(DB_PATH, "wb") as f:
                pickle.dump(db_data, f)
            self.db = db_data
            return len(chunks_data), "성공적으로 벡터 DB를 구축하였습니다."
        except Exception as e:
            return 0, f"벡터 DB 저장 실패: {e}"

    def search_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """사용자 쿼리와 가장 유사한 텍스트 청크를 검색합니다."""
        if not self.db:
            print("벡터 DB가 로드되지 않았습니다.")
            return []

        try:
            # 로컬 모델 쿼리 임베딩
            embedder = get_local_embedder()
            query_embedding = embedder.encode([query])[0]
            query_embedding = np.array(query_embedding, dtype=np.float32)
        except Exception as e:
            print(f"쿼리 임베딩 실패: {e}")
            return []

        # 코사인 유사도 연산 (NumPy 기반 고속 행렬 연산)
        embeddings = self.db["embeddings"]
        metadata = self.db["metadata"]

        dots = np.dot(embeddings, query_embedding)
        norms_db = np.linalg.norm(embeddings, axis=1)
        norm_q = np.linalg.norm(query_embedding)
        
        similarities = dots / (norms_db * norm_q + 1e-8)
        
        # 상위 top_k 인덱스 추출
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": metadata[idx]["text"],
                "source": metadata[idx]["source"],
                "page": metadata[idx]["page"],
                "similarity": float(similarities[idx])
            })
            
        return results

    def generate_rag_answer(self, query: str, test_results: Dict[str, Any], top_k: int = 3) -> str:
        """논문 근거 및 검사 결과에 기반하여 답변을 생성합니다."""
        anxiety_terms = ["죽", "큰일", "어떻게 되는", "무서", "위험한가", "위험해", "살 수"]
        if any(term in query for term in anxiety_terms):
            return (
                "아니요. 이 검사 결과만 보고 “곧 죽는다”거나 “큰일 났다”고 말할 수는 없습니다.\n\n"
                "이 검사는 병을 확정하는 검사가 아닙니다. "
                "후각, 배변, 손그림에서 주의가 필요한 신호가 있는지 살펴보는 참고용 검사입니다.\n\n"
                "지금 할 일은 너무 겁먹는 것이 아닙니다. "
                "결과 화면을 보호자나 보건소 담당자에게 보여주시고, "
                "증상이 계속되면 신경과 전문의에게 상담을 받아보시면 됩니다.\n\n"
                "다만 갑자기 심한 어지럼, 한쪽 팔다리 마비, 말이 어눌해짐, 의식 저하, "
                "가슴 통증이나 숨참이 있으면 파킨슨 검사와 별개로 119에 바로 연락해 주세요."
            )

        if not self.is_api_key_valid():
            return (
                "지금은 AI 상담 연결 설정을 확인해야 합니다.\n\n"
                "검사 결과 화면에 보이는 내용은 정상적으로 계산되었습니다. "
                "이 결과는 진단이 아니라 참고용입니다.\n\n"
                "걱정되는 증상이 계속되면 보호자나 보건소 담당자에게 결과를 보여주시고, "
                "신경과 전문의와 상담해 주세요."
            )

        # 유사 논문 청크 검색
        similar_chunks = []
        if self.db:
            similar_chunks = self.search_similar_chunks(query, top_k=top_k)
        
        # 검색된 근거 포맷팅
        context_str = ""
        if similar_chunks:
            for i, chunk in enumerate(similar_chunks, 1):
                context_str += f"[근거 {i}] 출처: {chunk['source']} (Page {chunk['page']})\n내용: {chunk['text']}\n\n"
        else:
            context_str = "검색된 논문 근거가 없습니다. 제공된 검사 결과만 간단히 설명하고, 구체적인 의학적 근거가 필요한 내용은 답변하지 말고 전문의 상담을 권장해 주세요."

        # 검사 결과 정보 포맷팅
        olf_score = sum(test_results.get("olf_results", []))
        scopa_score = sum(test_results.get("scopa_results", []))
        scopa_details = test_results.get("scopa_results", [])
        def pct_or_pending(value):
            return "아직 산출 전" if value is None else f"{float(value) * 100:.1f}%"

        p_olf = test_results.get("p_olf")
        p_img = test_results.get("p_img")
        p_kin = test_results.get("p_kin")
        final_risk = test_results.get("final_risk")

        results_str = (
            f"- 종합 파킨슨 위험도: {pct_or_pending(final_risk)}\n"
            f"- 후각 인지 점수: {olf_score} / 12\n"
            f"- 배변 불편 점수: {scopa_score} / 9 (높을수록 불편이 큼)\n"
            f"- 후각+배변 모델 주의 신호: {pct_or_pending(p_olf)}\n"
            f"  * Q5(변비 여부): {scopa_details[0] if len(scopa_details) > 0 else '미검사'}점\n"
            f"  * Q6(대변 안간힘): {scopa_details[1] if len(scopa_details) > 1 else '미검사'}점\n"
            f"  * Q7(변지림): {scopa_details[2] if len(scopa_details) > 2 else '미검사'}점\n"
            f"- 나선 그리기 이미지 위험 확률: {pct_or_pending(p_img)}\n"
            f"- 나선 그리기 속도/떨림(운동학) 위험 확률: {pct_or_pending(p_kin)}\n"
        )

        # 시스템 프롬프트 구성
        system_prompt = (
            "당신은 파킨슨병 조기 선별 검사 결과를 환자나 보호자에게 설명해주는 따뜻하고 신뢰감 있는 신경과 전문 AI 어시스턴트 '박인순'입니다.\n"
            "노인 사용자가 읽는다는 점을 최우선으로 생각하세요.\n"
            "항상 한국어 해요체로, 짧고 쉬운 문장을 사용하세요.\n"
            "답변 원칙:\n"
            "1. 전문용어는 가능한 쉬운 말로 바꿔 설명합니다.\n"
            "2. 한 문장은 짧게 작성합니다.\n"
            "3. 사용자가 불안해하지 않도록 단정적인 진단 표현을 피합니다.\n"
            "4. “파킨슨병입니다”라고 말하지 않습니다. “주의가 필요한 신호가 보입니다”처럼 설명합니다.\n"
            "5. 검사 결과는 참고용입니다. 정확한 진단은 신경과 전문의 상담이 필요하다고 안내합니다.\n"
            "6. Grad-CAM은 “AI가 그림에서 중요하게 본 부분을 색으로 표시한 것”이라고 설명합니다.\n"
            "7. 고위험, 중위험, 저위험 결과에 따라 다음 행동을 쉽게 안내합니다.\n"
            "8. 보호자나 보건소 담당자에게 보여줄 수 있는 요약도 함께 제공합니다.\n\n"
            "금지 표현:\n"
            "- 파킨슨병입니다\n"
            "- 파킨슨병이 확실합니다\n"
            "- 앞으로 파킨슨병에 걸립니다\n\n"
            "답변 구조:\n"
            "[1] 결과 요약\n"
            "[2] 쉬운 해석\n"
            "[3] 다음 행동\n"
            "[4] 보호자/보건소 담당자용 요약\n"
            "논문 근거는 내부 참고용으로만 사용하세요.\n"
            "답변에는 [5] 참고한 논문 근거, 근거 번호, 파일명, Page 번호를 표시하지 마세요.\n"
        )

        # 사용자 프롬프트 구성
        user_prompt = (
            f"[사용자 검사 결과]\n{results_str}\n"
            f"[로컬 논문 검색 근거]\n{context_str}\n"
            f"[사용자 질문]\n{query}\n\n"
            "검사 결과와 관련 논문 근거를 바탕으로 사용자의 질문에 친절하게 답변해 주세요. "
            "단, 논문 근거 목록과 출처 표기는 답변에 쓰지 마세요."
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception:
            return (
                "지금은 AI 상담 연결이 잠시 원활하지 않습니다.\n\n"
                "검사 결과 화면에 보이는 내용은 정상적으로 계산되었습니다. "
                "이 결과는 진단이 아니라 참고용입니다.\n\n"
                "걱정되는 증상이 계속되면 보호자나 보건소 담당자에게 결과를 보여주시고, "
                "신경과 전문의와 상담해 주세요."
            )

# -------------------------------------------------------
# CLI 실행부 (로컬 벡터 DB 생성 스크립트 역할)
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parkinson RAG DB Builder")
    parser.add_argument("--build", action="store_true", help="논문 PDF들을 기반으로 벡터 DB 구축")
    parser.add_argument("--query", type=str, help="테스트 쿼리 검색")
    
    args = parser.parse_args()
    
    if args.build:
        print("=== Parkinson RAG 로컬 벡터 DB 구축 시작 ===")
        rag = ParkinsonRAG()
        count, msg = rag.build_db_from_pdfs()
        print(f"결과: {msg} (총 {count}개 텍스트 청크 임베딩 완료)")
        
    elif args.query:
        rag = ParkinsonRAG()
        if not rag.db:
            print("벡터 DB를 찾을 수 없습니다. --build 옵션으로 먼저 구축해 주세요.")
            exit(1)
            
        print(f"쿼리 '{args.query}'에 대한 로컬 논문 검색 결과:")
        results = rag.search_similar_chunks(args.query, top_k=3)
        for i, res in enumerate(results, 1):
            print(f"\n[{i}] 출처: {res['source']} (Page {res['page']}), 유사도: {res['similarity']:.4f}")
            print(f"내용: {res['text']}")
