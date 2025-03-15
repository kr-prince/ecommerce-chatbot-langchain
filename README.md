# ecommerce-chatbot-langchain
Code for Fractal-Nexus GenAI hackathon for March 2025


### About

* The *data* folder has the all policy related information and customer orders data specific to the app in JSON and dB format.

* The *docs* folder has the problem statement and evaluation criteria for this hackathon.

* The *notebooks* folder has all the experimental code in notebook format

* The *utils* package has modules containing all the functions used to drive the app

* The *Nexus_Hackathon_Mar25.pdf* file is the project report document.


### Setup

* Clone the repo locally using `git clone https://github.com/kr-prince/ecommerce-chatbot-langchain.git`

* Look out for the `.env` file in `./data` folder and fill up the required Groq, Huggingface and Pinecone API keys.

* Launch Anaconda base environment and create the required environment by `conda env create -f environment.yml`. Activate the environment using `conda activate nexusEnv`. 

* Launch the chatbot app using `python app.py`

* Optionally, in separate terminals, run `python order_data_service.py` to start the orders data ingestion service and run `python policy_ingestion_service.py` to start the policy documents information ingestion service.

