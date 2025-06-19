import streamlit as st
from PIL import Image
from ragchain_langchain import qa_chain  # must expose qa_chain
from retriever import retrieve_chunks
# if needed

# Streamlit page setup
st.set_page_config(page_title="🌾Rice Farming Agent", layout="wide")
img_path = os.path.join(os.path.dirname(_file_), "Rice_Farming.png")
img = Image.open(img_path)
st.markdown("## 🌾 Rice Farming Assistance Agent")


# Banner

# Init session
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# Display all previous messages
for entry in st.session_state.chat_log:
    role = entry["role"]
    avatar = "🧑‍🌾" if role == "user" else "🤖🚜"
    st.markdown(f"**{avatar} {role.capitalize()}:**\n\n{entry['content']}")

# Input box
query = st.chat_input("Ask about rice farming...")
if query:
    # Show user message
    st.chat_message("user").markdown(query)
    st.session_state.chat_log.append({"role": "user", "content": query})

    # Run RAG chain
    result = qa_chain.invoke({"question": query})
    answer = result["answer"]
    docs = result.get("source_documents", [])

    # Show assistant response
    st.chat_message("assistant").markdown(answer)
    st.session_state.chat_log.append({"role": "assistant", "content": answer})

    # === Expanders: Sources and Chunks
    if docs:
        with st.expander("📚 Sources used"):
            shown = set()
            for doc in docs:
                meta = doc.metadata
                title = meta.get("title", "No title")
                url = meta.get("url", "N/A")

                # Avoid duplicate titles+urls
                if (title, url) not in shown:
                    shown.add((title, url))
                    st.markdown(f"- **{title}**\n  🔗[Open Source]({url})")

        with st.expander("📄 Chunks retrieved"):
            for doc in docs:
                meta = doc.metadata
                title = meta.get("title", "No title")
                score = meta.get("score", "?")
                chunk_id_full = meta.get("chunk_id", "N/A")
                chunk_id = chunk_id_full.split("_")[-1] if "chunk" in chunk_id_full else chunk_id_full

                st.markdown(
                    f"### 🧩 {title}"
                    f"\n- **Chunk:** `{chunk_id}` | **Score:** `{score:.3f}`\n\n"
                    f"_Summary: {meta.get('summary', '')}_\n\n"
                    f"**Preview:** {doc.page_content[:200]}..."
                )

