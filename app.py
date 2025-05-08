import streamlit as st
import torch
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from peft import PeftModel, PeftConfig

# Set page configuration
st.set_page_config(
    page_title="DrugLLM Chat",
    page_icon="💊",
    layout="centered"
)

# App title and description
st.title("💊 DrugLLM Chat")
st.markdown("Ask questions about prescription medications and get informative answers.")

# Model loading function
@st.cache_resource
def load_model():
    """Load the fine-tuned model and tokenizer."""
    # Model paths
    BASE_MODEL_NAME = "openai-community/gpt2"
    FINETUNED_MODEL_PATH = "models/drug-llm/final"
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(FINETUNED_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load the PEFT model with LoRA adaptors
    try:
        config = PeftConfig.from_pretrained(FINETUNED_MODEL_PATH)
        model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading PEFT model: {e}")
        model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_PATH)
    
    # Set device and move model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device

# Generate response function
def generate_response(instruction, model, tokenizer, device, 
                     max_length=150, temperature=0.7, top_p=0.9):
    """Generate a response to a given instruction."""
    # Format the input as in the training data
    input_text = f"Instruction: {instruction}\nResponse:"
    
    # Tokenize input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Extract only the response part
    try:
        response = generated_text.split("Response:")[1].strip()
    except IndexError:
        response = generated_text.strip()
        
    return response

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a welcome message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hello! I'm DrugLLM, your medication information assistant. Ask me any questions about prescription drugs."
    })

# Load model (will use cached version after first load)
with st.spinner("Loading the DrugLLM model..."):
    model, tokenizer, device = load_model()
    st.success("Model loaded successfully!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Model parameters in sidebar
with st.sidebar:
    st.header("Model Parameters")
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1,
                          help="Higher values make output more random, lower values more deterministic")
    max_length = st.slider("Max Response Length", min_value=50, max_value=500, value=150, step=50,
                         help="Maximum number of tokens in the generated response")
    top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.9, step=0.1,
                     help="Nucleus sampling parameter")
    
    st.markdown("---")
    st.markdown("### About DrugLLM")
    st.markdown("This chatbot uses a fine-tuned language model specialized in prescription drug information.")
    st.markdown("Note: This is for informational purposes only and not a substitute for professional medical advice.")

# Chat input
if prompt := st.chat_input("Ask a question about medications..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(
                prompt, model, tokenizer, device,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
        st.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add example questions
with st.expander("Example Questions"):
    examples = [
        "What are the common side effects of Lisinopril?",
        "How should I take Metformin?",
        "What is Atorvastatin used for?",
        "Can you explain how Sertraline works?",
        "What should I know before taking Amlodipine?",
        "Are there any drug interactions with Levothyroxine?"
    ]
    
    for example in examples:
        if st.button(example):
            # Add to chat input (this will trigger the chat input callback)
            st.session_state.messages.append({"role": "user", "content": example})
            with st.chat_message("user"):
                st.write(example)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_response(
                        example, model, tokenizer, device,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p
                    )
                st.write(response)
            
            # Add to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()