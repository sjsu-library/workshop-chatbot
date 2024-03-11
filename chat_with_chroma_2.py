import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import chromadb

st.set_page_config(page_title="Kingbot - SJSU Library", page_icon="🤖")
st.title("Chat with SJSU Library's Kingbot 🤖")
st.text("This experimental version of Kingbot uses Streamlit, LangChain, and ChatGPT.")
@st.cache_resource(ttl="1h")
def configure_retriever():
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    client = chromadb.HttpClient(host=st.secrets["host"], port=8000)
    dbremote = Chroma(
        client=client,
        collection_name="sjsulib",
        embedding_function=embedding,
    )
    retriever = dbremote.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 5})
    return retriever
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None
    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)
class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")
    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")
    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

openai_api_key = st.secrets["key"]

retriever = configure_retriever()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True)

template = ''' Use the following pieces of context to answer the user's question:
{context}
Question: {question}
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If possible, end your response with a url to a relevant resource from Document's metadata.
'''
prompt = ChatPromptTemplate.from_template(template)

chat_history = []
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key='answer'
)
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, 'document_variable_name': 'context'},
    memory=memory
)
avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)
if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain(user_query, callbacks=[retrieval_handler, stream_handler])
        answer = response['answer']
        st.write(answer)
