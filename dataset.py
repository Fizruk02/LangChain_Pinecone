from include import *

#loader = UnstructuredPDFLoader("start.pdf")
loader = OnlinePDFLoader("https://3mu.ru/wp-content/uploads/2021/09/skazka-kolobok-a4.pdf")
data = loader.load()

print (f'Data: {len(data)}')
print (f'Characters: {len(data[0].page_content)}')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(data)

print (f'Documents: {len(texts)}')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
docsearch = Pinecone.from_texts([t.page_content for t in texts], index_name=INDEX_NAME)