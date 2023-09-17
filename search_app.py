from flask import Flask, jsonify, request
from search_engine import *
from config.config_parser import Config


# Import configurations from config.yml
conf = Config('config/config.yml')

# Assign configuration parameters to variables
FILEDIR = conf.configurations.file_dir
MODELNAME = conf.configurations.model_name
COLUMN = conf.configurations.col

# Call create dataframe and generate embeddings functions from search_engine
df = read_data(FILEDIR)
model, embeds = encode_corpus(df, MODELNAME, COLUMN)

app = Flask(__name__)

# Flask REST API to return top results
@app.route('/sementic', methods=['GET'])
def sementic():
    """
    API parameters:
    ---------------
    @param user_query: User input query
    @param results: Number of top results needs to be returned
    API body: 
            {
                "user_query": "Where can I buy my own wallbox for charging at home?",
                "results": 2
            }

    API ouput:
    ----------
    @return Number of top results
    """

    # Validate if the request is GET
    if request.method != "GET":
        return jsonify(code=1, response="Invalid request type"), 400

    # Extract API parameters
    query = request.get_json()
    
    # Assign API parameters into variables
    try:
        data = query['user_query']
        results = query['results']

    except KeyError:
        return jsonify(code=2, response="Invalid data keys"), 400
    
    # Return search results
    try:
        _, I = search_vect([data], model, embeds, df, num_results=results)
        return jsonify(get_results(df, I, COLUMN)), 200

    except Exception as e:
        return jsonify(e), 403


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)