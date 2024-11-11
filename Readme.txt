###### Network Security - Group 1 #######

Project Description:
The goal is this project is to develop a Q&A bot which answers the network security course questions while maintaining security by using private and local data. In this project, we are going to use GPT-4-ALL for language processing and Qdrant for storing the data efficiently. By integrating all these, the chatbot that we are going to develop answers accurately using the lecture slides, textbooks, and other course materials. If the answers are not found from the given lectures, then the bot will cite the internet  sources by running on a local machine. In addition , Wireshark is used to capture the data while working on the step 4 of LLM workflow.
----------------------------------------------------------------------------------------
System Architecture:
----------------------------------------------------------------------------------------
Pre-requisites:
Python: Version 3.9
Gradio: Web framework for Python
Git: version Control
Docker : Database control
----------------------------------------------------------------------------------------
Requirements:
1.Python Libraries:
 Install the following libraries using following commands,
	pip install gpt4all                                        
	pip install sentence transformers
	pip install qdrant_client
	pip install PyMuPDF (for “fitz”)
	pip install gradio

2.Environment Variables:
Set  SERPAPI_API_KEY as an environment variable taking SERPAPI_KEY from the SERP.AI website.

3.Adopted libraries: (Install below libraries using pip install command)
a. sentencetransformers : Embedding model of the project
b. qdrant_client : Connections to the qdrant database
c. gradio : Front-end web page
d. gpt4all : LLM model package of the project (used Meta-Llama-3 for better responses , however previously tested with Orca-mini which has provided faster responses due to small size of the model)
e. fitz : conversion of pdf files to text files to support use in sentence transformers

4. Docker Setup - qdrant Vector Database Setup

-> Install Docker and enable terminal
-> Run Below commands to start the qdrant Vector Database
	1. docker pull qdrant/qdrant # download qdrant into docker
	2. docker run -p 6333:6333 qdrant/qdrant # qdrant instance will be started on port 6333

5. Knowledge Database:
-> collect various lectures and papers and store them in the folder
----------------------------------------------------------------------------------------
Step by step instructions for execution:
a. Install Docker, enable terminal and create Qdrant instance
b. Run the Qdrant instance 
c. copy all the knowledge documents into a folder
d. Run initialise_qdrant.py to create a new collection in the qdrant DB
e. Place all the knowledge documents in one path and modify the pdf_path in the Data_insertion_qdrant.py
f. Run Data_insertion_qdrant.py which will load each page into the qdrant DB which will help in actively pointing referencing the document and page number.
g. Run Chatbot_application.py to start the local host and connection to qdrant DB, the local host is used to input query and receive the response from the vector DB if present or else connect to internet via a API to collect the data required.
	Place this url address http://127.0.0.1:7860 into the browser then
a. The user inputs a question into the interface.
b. The system converts the user's query into an embedding using the SentenceTransformers model.
c. The Qdrant database compares the query embedding to its stored document embeddings and returns the most relevant matches.
d. These document excerpts, along with the original question, are provided to GPT4All, which generates a context-aware response.
e. The bot presents the answer along with citations from the relevant course materials, ensuring transparency and academic integrity.
----------------------------------------------------------------------------------------
Video: Video has been uploaded to the github repositary to show how chatbot is working and shows the packet transfer in the wireshark as well.
----------------------------------------------------------------------------------------
Features:
a. Understanding and answering questions
•	The chatbot understands the prompts given by the user and gives the answers based on the a.
b. Citations and web references
•	The chatbot includes the page number, document names along with answer in the chatbot  response ,if the data is pulled  from the internet it will provide the web references.
c. Data Security and privacy
•	Since the chatbot runs on local machine, it ensures the data security and privacy.
d. Offline functionality
•	The bot answers the prompt even without internet.
----------------------------------------------------------------------------------------
Issues we experienced and the solutions:
a. Pdf parsing:
Issue:
•	If the data(lectures) is not structured properly then the chatbot might not correctly read that which results in incorrect response.
Solution:
•	Clean the data before feeding it to the model
b. latency:
Issue:
•	While extracting the data and generating documents there may be a chance of latency.
Solution:
•	Indexing documents and storing embeddings in Qdrant.
----------------------------------------------------------------------------------------
Suggestions for future improvement:

a. Enhancing UI design:
•	Integrating features like follow-up question and custom responses.
b. Performance optimization:
•	Optimize the performance of the chatbot to reduce the load times. 
c. Ambiguity handling:
•	Improve the bot to understand the ambiguous queries also.
----------------------------------------------------------------------------------------
Feedback:
a. Offline functionality: The bot runs without internet also.
b. Data security and Privacy: The Chatbot runs on the local machine ,ensures data security abd privacy
c. Citations: The bot provide the citations if it takes information from slides and provides web reference if the information pulled from internet.
----------------------------------------------------------------------------------------



