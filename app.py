import re
import json
import warnings
import time
import streamlit as st
from chain import load_chain, process_messages, question_relatable

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

model = "gpt-4o"

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

sidebar_content = """
<h1 id="heading">Clinical Trials Bot</h1>
<p>Welcome to Clinical Trials Bot, your intuitive companion for exploring clinical trials data through natural language queries. Whether you're seeking specific study details or general insights, it makes information retrieval effortless and engaging.</p>

<h2 id="heading">Features</h2>
<ul>
    <li><strong>Conversational AI:</strong> Harnessing ChatGPT to interpret natural language queries accurately</li>
    <li><strong>Conversational Memory:</strong> Retains context for interactive, dynamic responses.</li>
    <li><strong>Dataset:</strong> Based on a comprehensive dataset of 10,000 clinical trials.</li>
    <li><strong>Interactive User Interface:</strong> Transforms data querying into an engaging conversation, complete with a chat reset option.</li>
</ul>

<h2 id="heading">Example Queries</h2>
<ul>
    <li>What is the purpose of the study? In what phase is the study?</li>
    <li>Who sponsors the study, and who has reviewed and approved it?</li>
    <li>What is the eligibility criteria for the study?</li>
    <li>What are the outcomes for this study?</li>
    <li>How often and for how long will I receive the treatment, and how long will I need to remain in the study?</li>
</ul>
<style>
#input-container { 
    position: fixed; 
    bottom: 0; 
    width: 100%; 
    padding: 10px; 
    background-color: white; 
    z-index: 100; 
} 

#heading{ 
    font-weight: bold; 
    background: -webkit-linear-gradient(left, red, orange); 
    background: linear-gradient(to right, red, orange); 
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
    display: inline; 
} 
</style>
"""

st.markdown(gradient_text_html, unsafe_allow_html=True)
st.sidebar.markdown(sidebar_content,unsafe_allow_html=True)

st.caption("Talk your way through data")

def show_messages(input, chat_history, result):

    if result:
        for idx, chunk in enumerate(chain.stream({"input": input, "chat_history": chat_history})):
            if idx >= 2:
                yield chunk["answer"]
    else:
        response = "Apologies, I'm unable to provide a response to that inquiry at this moment. For further assistance, please feel free to reach out to us via phone at 714-456-7890 or visit our website at ucihealth.org. We'll be happy to help you there."
        for word in response.split():
            yield word + " "
            time.sleep(0.05)
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

    if st.sidebar.button("Reset Chat", type = "primary"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state["messages"] = INITIAL_MESSAGE
        st.session_state["history"] = []
        st.rerun()


    st.sidebar.markdown(
        "**Note:** <span style='color:red'>The bot's training data is limited to a dataset of 10,000 clinical trials.</span>",
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state.keys():
        st.session_state["messages"] = INITIAL_MESSAGE
    
    if "history" not in st.session_state.keys():
        st.session_state["history"] = []

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "user", "content": prompt})

    for message in st.session_state["messages"] :
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])


    chain, vector_store = load_chain(model, "retrieval_chain")
    if ("messages" in st.session_state and st.session_state["messages"][-1]["role"] != "assistant"):

        user_input_content = st.session_state["messages"][-1]["content"]
        if isinstance(user_input_content, str):

            with st.chat_message("assistant"):
                botmsg = st.empty()

            chat_history = process_messages(st.session_state.history)
            print(f"{chat_history} \n")
            result = question_relatable(chat_history, user_input_content)
            ai_msg = botmsg.write_stream(show_messages(user_input_content, chat_history, result))
            
            if result:
                st.session_state.history.append({'role': "assistant", 'content': ai_msg})
            else:
                st.session_state.history.pop()

            print(st.session_state.history)
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