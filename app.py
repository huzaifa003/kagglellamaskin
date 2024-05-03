from llama_index import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:26:17.105325Z","iopub.execute_input":"2024-05-03T18:26:17.106074Z","iopub.status.idle":"2024-05-03T18:26:22.131253Z","shell.execute_reply.started":"2024-05-03T18:26:17.106042Z","shell.execute_reply":"2024-05-03T18:26:22.130395Z"}}
documents=SimpleDirectoryReader("dermbook/").load_data()
# documents

# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:26:22.132981Z","iopub.execute_input":"2024-05-03T18:26:22.133540Z","iopub.status.idle":"2024-05-03T18:26:22.138972Z","shell.execute_reply.started":"2024-05-03T18:26:22.133502Z","shell.execute_reply":"2024-05-03T18:26:22.138026Z"}}
system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible prefferably based on the instructions and context provided,
if the context doesn't have exact definition, try to match the questions that suit the context provided,
if there is any question apart from skin, it's diseases, it's treatments, say that it's out of context and can't be answered
"""
## Default format supportable by LLama2
query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:26:22.140267Z","iopub.execute_input":"2024-05-03T18:26:22.140577Z","iopub.status.idle":"2024-05-03T18:27:46.638823Z","shell.execute_reply.started":"2024-05-03T18:26:22.140548Z","shell.execute_reply":"2024-05-03T18:27:46.637982Z"}}
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import ServiceContext
from llama_index.embeddings import LangchainEmbedding

embed_model=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh"))

# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:27:46.641585Z","iopub.execute_input":"2024-05-03T18:27:46.642501Z","iopub.status.idle":"2024-05-03T18:29:29.760891Z","shell.execute_reply.started":"2024-05-03T18:27:46.642471Z","shell.execute_reply":"2024-05-03T18:29:29.760145Z"}}
import torch

llm = HuggingFaceLLM(
    context_window=2056,
    max_new_tokens=256,
    generate_kwargs={"temperature" : 0.0},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    # uncomment this if using CUDA to reduce memory usage
    #model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)


# %% [code]


# %% [code]


# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:29:29.761922Z","iopub.execute_input":"2024-05-03T18:29:29.762193Z","iopub.status.idle":"2024-05-03T18:29:31.787684Z","shell.execute_reply.started":"2024-05-03T18:29:29.762170Z","shell.execute_reply":"2024-05-03T18:29:31.786878Z"}}
service_context=ServiceContext.from_defaults(
    chunk_size=8096,
    llm=llm,
    embed_model=embed_model
)


# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:29:31.788766Z","iopub.execute_input":"2024-05-03T18:29:31.789241Z","iopub.status.idle":"2024-05-03T18:29:37.634407Z","shell.execute_reply.started":"2024-05-03T18:29:31.789210Z","shell.execute_reply":"2024-05-03T18:29:37.633298Z"}}
index=VectorStoreIndex.from_documents(documents,service_context=service_context)

# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:29:37.635588Z","iopub.execute_input":"2024-05-03T18:29:37.635937Z","iopub.status.idle":"2024-05-03T18:29:38.137549Z","shell.execute_reply.started":"2024-05-03T18:29:37.635903Z","shell.execute_reply":"2024-05-03T18:29:38.136605Z"}}
query_engine=index.as_query_engine(streaming=True)

# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:29:38.138743Z","iopub.execute_input":"2024-05-03T18:29:38.139522Z","iopub.status.idle":"2024-05-03T18:30:51.643576Z","shell.execute_reply.started":"2024-05-03T18:29:38.139483Z","shell.execute_reply":"2024-05-03T18:30:51.642741Z"}}

response=query_engine.query("What are some treatments of Eczema?")

for text in response.response_gen:
    # do soething with text as they arrive.
    print(text.strip())

# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:25:19.428352Z","iopub.status.idle":"2024-05-03T18:25:19.428780Z","shell.execute_reply.started":"2024-05-03T18:25:19.428557Z","shell.execute_reply":"2024-05-03T18:25:19.428575Z"}}



# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:25:19.430151Z","iopub.status.idle":"2024-05-03T18:25:19.430471Z","shell.execute_reply.started":"2024-05-03T18:25:19.430316Z","shell.execute_reply":"2024-05-03T18:25:19.430329Z"}}
response

# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:30:51.644832Z","iopub.execute_input":"2024-05-03T18:30:51.645486Z","iopub.status.idle":"2024-05-03T18:30:51.650238Z","shell.execute_reply.started":"2024-05-03T18:30:51.645449Z","shell.execute_reply":"2024-05-03T18:30:51.649287Z"}}
import os

# Path to the ngrok executable
ngrok_executable = '/kaggle/working/pyngrok/bin/ngrok'

# Change the mode of the executable to add execute permissions
os.chmod(ngrok_executable, 0o755)  # This sets the file to be readable and executable by everyone


# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:30:51.653871Z","iopub.execute_input":"2024-05-03T18:30:51.654166Z","iopub.status.idle":"2024-05-03T18:30:51.673427Z","shell.execute_reply.started":"2024-05-03T18:30:51.654142Z","shell.execute_reply":"2024-05-03T18:30:51.672616Z"}}
from pyngrok import ngrok

# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:30:51.674531Z","iopub.execute_input":"2024-05-03T18:30:51.674925Z","iopub.status.idle":"2024-05-03T18:30:51.678990Z","shell.execute_reply.started":"2024-05-03T18:30:51.674894Z","shell.execute_reply":"2024-05-03T18:30:51.678136Z"}}
ngrok_key = "2f3yvUnCJ6EnQKtJZqmsgSZhXvN_2dGME25rX74p7MtEmzTcT"
port = 5000

# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:30:51.680201Z","iopub.execute_input":"2024-05-03T18:30:51.680478Z","iopub.status.idle":"2024-05-03T18:30:52.713801Z","shell.execute_reply.started":"2024-05-03T18:30:51.680455Z","shell.execute_reply":"2024-05-03T18:30:52.712838Z"}}
ngrok.set_auth_token(ngrok_key)
ngrok.connect(port, domain='bursting-allegedly-fawn.ngrok-free.app').public_url

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:30:52.715401Z","iopub.execute_input":"2024-05-03T18:30:52.715754Z","iopub.status.idle":"2024-05-03T18:30:52.991950Z","shell.execute_reply.started":"2024-05-03T18:30:52.715710Z","shell.execute_reply":"2024-05-03T18:30:52.991042Z"}}
from flask import Flask, request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# %% [code] {"execution":{"iopub.status.busy":"2024-05-03T18:30:52.992983Z","iopub.execute_input":"2024-05-03T18:30:52.993575Z"}}
from flask import Flask, request, Response, jsonify

# app = Flask(__name__)

@app.route("/derm", methods=['POST'])
def derm():
    try:
        # Assuming the data is sent as JSON, with a key named 'query'
        data = request.json
        query = data.get('query', 'Name some common skin diseases')
        print(query)

        # Simulating a query engine call
        response = query_engine.query(query)  # Use the query from the user

        def generate():
            # Assuming `streaming_response.response_gen` is your generator
            # Replace with the actual generator you are using
            for text in response.response_gen:
                # Process text here, e.g., format or simply yield
                yield text + '\n'  # Adding a newline for proper streaming

        # Create a streaming response with the generator and set the appropriate MIME type
        return Response(generate(), mimetype='text/plain')

    except Exception as e:
        # Handle generic exceptions, which captures any error that wasn't explicitly caught
        error_message = {"error": str(e)}
        return jsonify(error_message), 500  # Return a JSON response with 500 Internal Server Error

if __name__ == '__main__':
    app.run(port=5000)  # Adjust the port if necessary