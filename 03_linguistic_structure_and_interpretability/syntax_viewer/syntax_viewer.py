import pandas as pd
import spacy
import streamlit as st
import torch
from spacy import displacy
from transformers import AutoModel, AutoTokenizer


st.set_page_config(page_title="Syntax and attention", layout="wide")
st.title("Syntax and attention")
st.write(
    "Compare a dependency analysis with one attention matrix. "
    "They describe different objects and should not be read as interchangeable explanations."
)


@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def load_transformer():
    model_id = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        output_attentions=True,
        attn_implementation="eager",
    )
    model.eval()
    return tokenizer, model


nlp = load_spacy_model()
tokenizer, model = load_transformer()

text = st.text_input(
    "English sentence",
    "The trophy did not fit in the suitcase because it was too large.",
)

if text:
    st.header("Dependency parsing")
    document = nlp(text)
    dependency_html = displacy.render(
        document,
        style="dep",
        options={"compact": True, "distance": 105, "bg": "#ffffff"},
    )
    st.write(dependency_html, unsafe_allow_html=True)

    st.header("Transformer attention")
    inputs = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)

    attentions = outputs.attentions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    layer_column, head_column = st.columns(2)
    layer = layer_column.slider("Layer", 0, len(attentions) - 1, len(attentions) - 1)
    head = head_column.slider(
        "Head",
        0,
        attentions[layer].shape[1] - 1,
        0,
    )

    attention_matrix = attentions[layer][0, head].cpu().numpy()
    attention_table = pd.DataFrame(
        attention_matrix,
        index=tokens,
        columns=tokens,
    )
    st.dataframe(
        attention_table.style.background_gradient(cmap="Blues", axis=None).format("{:.2f}"),
        use_container_width=True,
    )
    st.caption(
        "Rows are source tokens and each row sums to one. A large weight shows routing "
        "inside this head; it is not a dependency edge or proof of a linguistic explanation."
    )
