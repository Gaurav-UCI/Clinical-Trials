import re
import json
import warnings
import streamlit as st
from chain import load_chain, process_messages

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

model = "gpt-4-turbo"

gradient_text_html = """
<style>
.gradient-text {
    font-weight: bold;
    background: -webkit-linear-gradient(left, red, orange);
    background: linear-gradient(to right, red, orange);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 3em;
}
</style>

<div class="gradient-text">Clinical Trials Bot</div>
"""
st.markdown(gradient_text_html, unsafe_allow_html=True)
st.caption("Talk your way through data")

try:
    if "toast_shown" not in st.session_state:
        st.session_state["toast_shown"] = False

    if "rate-limit" not in st.session_state:
        st.session_state["rate-limit"] = False


    if not st.session_state["toast_shown"]:
        st.toast("The bot's training data is limited to a dataset of 10,000 clinical trials.", icon="👋")
        st.session_state["toast_shown"] = True

    if st.session_state["rate-limit"]:
        st.toast("Probably rate limited.. Go easy folks", icon="⚠️")
        st.session_state["rate-limit"] = False


    INITIAL_MESSAGE = [
        {"role": "user", "content": "Hi! 👋"},
        {
            "role": "assistant",
            "content": "Hello! I'm here to provide information and assistance regarding clinical trials. How can I help you today?"
        },
    ]

    with open("./content/sidebar.md") as file:
        sidebar_content = file.read()

    with open("./content/styles.md") as file:
        styles_content = file.read()

    st.sidebar.markdown(sidebar_content, unsafe_allow_html=True)

    if st.sidebar.button("Reset Chat", type = "primary"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state["messages"] = INITIAL_MESSAGE


    st.sidebar.markdown(
        "**Note:** <span style='color:red'>The bot's training data is limited to a dataset of 10,000 clinical trials.</span>",
        unsafe_allow_html=True,
    )

    st.write(styles_content, unsafe_allow_html=True)

    if "messages" not in st.session_state.keys():
        st.session_state["messages"] = INITIAL_MESSAGE

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state["messages"] :
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])


    chain = load_chain(model, "retrieval_chain")

    if ("messages" in st.session_state and st.session_state["messages"][-1]["role"] != "assistant"):

        user_input_content = st.session_state["messages"][-1]["content"]

        if isinstance(user_input_content, str):

            with st.chat_message("assistant"):
                botmsg = st.empty()
            
            chat_history = process_messages(st.session_state.messages)
            with st.spinner("Thinking..."):
                ai_msg = chain.invoke({"input": user_input_content, "chat_history": chat_history})["answer"]
            
            botmsg.write(ai_msg)

            st.session_state.messages.append({"role": "assistant", "content": ai_msg})


except Exception as e:
    print(e)
    st.error("An unexpected error occured. Please try again later! You can refresh the application using the button provided and or try again at later stage!")
    col1, col2, col3 = st.columns(3)

    with col2:
        if st.button("Reset Application", type="primary"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.session_state["messages"] = INITIAL_MESSAGE
            st.rerun()