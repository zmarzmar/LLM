import streamlit as st
from dotenv import load_dotenv  # API KEY를 환경변수로 관리하기 위한 설정 파일
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate  # 프롬프트 템플릿
from langchain_openai import ChatOpenAI  # llm
from langchain_core.output_parsers import StrOutputParser  # Output Parser
from langchain_core.prompts import load_prompt  # 프롬프트 로드
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote import logging
import os

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project]PDF RAG")

load_dotenv()  # API KEY 정보로드

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("PDF 기반 QA")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록 저장 용도
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

# 사이드바
with st.sidebar:
    # 초기화 버튼
    clear_btn = st.button("대화 초기화")
    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])
    # 모델 선택 메뉴
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    )


# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 이전 메시지 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever


# 체인 생성
def create_chain(retriever, model_name="gpt-4o"):
    # prompt = load_prompt(prompt_filepath, encoding="utf-8")
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if uploaded_file:
    # 파일 업로드 후 retriever 생성(작업 시간이 오래 걸릴 예정)
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain


# 초기화 버튼을 눌렀을 때
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 내용 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 사용자의 입력이 들어오면
if user_input:
    # 체인 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 웹에 대화 출력(이모지 사용)
        st.chat_message("user").write(user_input)

        # 지피티처럼 답변이 계속해서 나오는 방법(스트리밍 호출)
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            container = st.empty()

            ai_answer = " "
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 답변이 나올때까지 기다리는 방법
        # ai_answer = chain.invoke({"question": user_input})
        # AI 답변
        # st.chat_message("assistant").write(ai_answer)

        # 대화기록 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일을 업도르 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")
