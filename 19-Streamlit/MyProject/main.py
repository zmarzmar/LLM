import streamlit as st
from dotenv import load_dotenv  # API KEY를 환경변수로 관리하기 위한 설정 파일
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate  # 프롬프트 템플릿
from langchain_openai import ChatOpenAI  # llm
from langchain_core.output_parsers import StrOutputParser  # Output Parser
from langchain_core.prompts import load_prompt  # 프롬프트 로드
from langchain import hub  # 허브

load_dotenv()  # API KEY 정보로드

st.title("나의 chatgpt")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록 저장 용도
    st.session_state["messages"] = []

# 사이드바
with st.sidebar:
    # 초기화 버튼
    clear_btn = st.button("대화 초기화")

    selected_prompt = st.selectbox(
        "프롬프트를 선택해주세요", ("기본모드", "SNS 게시글", "요악"), index=0
    )


# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 이전 메시지 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 체인 생성
def create_chain(prompt_type):
    # prompt | llm | output_parser

    # prompt(기본모드)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신을 친절한 AI 어시스턴스 입니다. 다음 질문에 간결하게 답변해 주세요.",
            ),
            ("human", "#Question:\n{question}"),
        ]
    )

    if prompt_type == "SNS 게시글":
        prompt = load_prompt("prompts/sns.yaml", encoding="utf-8")
    elif prompt_type == "요약":
        prompt = hub.pull("teddynote/chain-of-density-map-korean")

    # llm
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 출력 파서
    output_parser = StrOutputParser()

    # 체인 생성
    chain = prompt | llm | output_parser
    return chain


# 초기화 버튼을 눌렀을 때
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 내용 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 사용자의 입력이 들어오면
if user_input:
    # 웹에 대화 출력(이모지 사용)
    st.chat_message("user").write(user_input)
    # 체인 생성
    chain = create_chain(selected_prompt)

    # 지피티처럼 답변이 계속해서 나오는 방법(스트리밍 호출)
    response = chain.stream({"question": user_input})
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
