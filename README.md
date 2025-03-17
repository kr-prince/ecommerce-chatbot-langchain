# ecommerce-chatbot-langchain
Code for Fractal-Nexus GenAI hackathon for March 2025


## About

* The *data* folder has the all policy related information and customer orders data specific to the app in JSON and dB format.

* The *docs* folder has the problem statement and evaluation criteria for this hackathon.

* The *notebooks* folder has all the experimental code in notebook format.

* The *utils* package has modules containing all the functions used to drive the app.

* The *Nexus_Hackathon_Mar25.pdf* file is the project report document.


## Prerequisite

* Free tier API keys of [Groq](https://console.groq.com/keys), [Pinecone](https://docs.pinecone.io/guides/get-started/overview) and [Huggingface](https://huggingface.co/).


## Setup

1. Clone the repo locally `git clone https://github.com/kr-prince/ecommerce-chatbot-langchain.git`

2. Look out for the `.env` file in `./data` folder and fill up the required Groq, Huggingface and Pinecone API keys.

3. Launch Anaconda base environment and create the required environment `conda env create -f environment.yml`

4. Activate the environment by `conda activate nexusEnv`

5. Check the `./utils/configs.py` file for default values and settings. Make changes only if anything specific is required.

6. Run `python order_data_service.py` to start the orders data ingestion service. This will use the `./data/orders_table.xlsx` file to create the `./data/chatbot.db` file when run for first time. It will keep checking for any additional order details every hour. Keeping it running is optional for bot functioning.

7. Run `python policy_ingestion_service.py` to start the policy information documents ingestion service. This works only after the orders data ingestion service. This processes each of the *.pdf* files in `./data/policy_docs/` folder to create a corresponding *.json* file. It also then uploads the data in Pinecone vector index and keeps checking for any additional data every hour. Keeping it running is optional for bot functioning.

8. Launch the main chatbot app by `python app.py`

9. Open the Gradio link and fire away..!!

