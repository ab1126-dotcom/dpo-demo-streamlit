import streamlit as st, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

USER = "abd87"
MODELS = {
    "Base (Qwen2.5-0.5B)": "Qwen/Qwen2.5-0.5B",
    "DPO beta=0.1": f"{USER}/qwen2.5-0.5b-dpo-beta0.1",
    "DPO beta=0.5": f"{USER}/qwen2.5-0.5b-dpo-beta0.5",
}

st.set_page_config(page_title="DPO Alignment Demo")
st.title("Base vs DPO-aligned Qwen2.5-0.5B")
st.caption("Choose a model, type a prompt, and see its response (greedy decoding, CPU).")

@st.cache_resource(max_entries=1)
def load(path):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).eval()
    return tok, model

name = st.selectbox("Model", list(MODELS))
prompt = st.text_area("Prompt", "Explain how the internet works to a 12-year-old.", height=100)
max_new = st.slider("Max new tokens", 50, 300, 150, 10)

if st.button("Generate"):
    with st.spinner("Loading model and generating (can take 1-2 minutes)..."):
        tok, model = load(MODELS[name])
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
        answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    st.subheader(name)
    st.write(answer)
