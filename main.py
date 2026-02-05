import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vector_store = PineconeVectorStore( embedding=embeddings, index_name=os.environ['INDEX_NAME'])

#LIMTIS TO ONLY 3 TOP DOCUMENTS WE ARE GOING TO USE
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    Provide a detailed answer:"""
)

print("Components Initialized.")

def format_docs(docs):
    """Format retrieved docs into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    query = "What is pinecone in machine learning?"
    """
    ===================================================
    OPTION 1. NOT USING LCEL 
    ===================================================
    """
    def retrieval_chain_without_lcel(query: str):
        """
        Simple retrieval chain without LCEL. Manually retrieves documents, formats them and generates a response.
        :param query: The query asked by the user.
        :return:
        """
        docs = retriever.invoke(query)
        context = format_docs(docs)
        messages = prompt_template.format_messages(context=context, question=query)
        response = llm.invoke(messages)
        return response.content


    # result_without_lcel = retrieval_chain_without_lcel(query)
    # print("/nAnswer:")
    # print(result_without_lcel)
    """
    ===================================================
    OPTION 2.  USING LCEL 
    ===================================================
    """
    def create_retrieval_chain_with_lcel():
        """
        Create a retrieval chain using LCEL (Langcahin Expression Language).
        Returns a chain that can be invoked with {"question" : "..."}
        :return:
        Returns an LCEL chain to invoke
        """

        #StrOutputParser will serve to access final .content from LLM response.
        #RunnablePassthrough converts our original function chain to a Runnable object in langchain, this makes it so
        #In this example, context will be assigned by langchain by running the subchain context= itemgetter[str]("question") | retriever | format_docs
        retrieval_chain = (
            RunnablePassthrough.assign(
                context= itemgetter("question") | retriever | format_docs
            )
            | prompt_template
            | llm
            | StrOutputParser()
        )
        return retrieval_chain

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("Answer with LCEL:\n")
    print(result_with_lcel)
