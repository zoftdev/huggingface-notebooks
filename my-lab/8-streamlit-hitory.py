import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize history
if 'history' not in st.session_state:
    st.session_state['history'] = ""

# Streamlit app layout
st.title("Chat with GPT-2")
st.write("Ask anything!")

# Input box for user question
user_input = st.text_input("You:", key="user_input")

# Display chat history
st.text_area("History:", value=st.session_state['history'], height=300)

# When user submits a question
if st.button("Send"):
    # Append user input to history
    st.session_state['history'] += "User: " + user_input + "\n"

    # Encode the new input with the past context
    input__ids = tokenizer.encode(st.session_state['history'], return_tensors='pt')

    # Generate a response from GPT-2
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Append response to history
    st.session_state['history'] += "Bot: " + response + "\n"

    # Show the updated history
    st.text_area("History:", value=st.session_state['history'], height=300)