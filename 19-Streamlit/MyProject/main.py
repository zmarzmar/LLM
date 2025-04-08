import streamlit as st
from langchain_core.messages.chat import ChatMessage
from dotenv import load_dotenv  # API KEY를 환경변수로 관리하기 위한 설정 파일
from langchain_core.prompts import ChatPromptTemplate  # 프롬프트 템플릿
from langchain_core.prompts import PromptTemplate  # llm
from langchain_core.output_parsers import StrOutputParser  # Output Parser

load_dotenv()  # API KEY 정보로드

st.title("나의 chatgpt test")

if "messages" not in st.session_state:
    # 대화기록 저장 용도
    st.session_state["messages"] = []


# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 이전 메시지 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 사용자의 입력이 들어오면
if user_input:
    # 웹에 대화 출력(이모지 사용)
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(user_input)

    # 대화기록 저장
    add_message("user", user_input)
    add_message("assistant", user_input)
