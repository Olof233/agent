import utils
import models
import tools
from flask import Flask, request, render_template, jsonify
import uuid
import threading
import json


app = Flask(__name__)
task_store = {}
contents = {}
path = "config/models.yaml"
config = utils.read_yaml(path)
models_list = utils.load_models_from_config(config)
# tool_definitions = tools.init_tools()


def main(content):
    conversations = []
    prompt={'role': 'system',
            'content': """'Evaluate the provided report or content, rating each of the following six dimensions on a scale from **0 to 10**. For each dimension, provide: 

            * **A numerical rating** (0–10).
            * **Brief justification** based directly on the toolcall results.
            * **Suggestions for improvement** (if rating is below 8).

            Dimensions for evaluation:

            1. **Language Correction**:
            Assess accuracy in spelling, grammar, punctuation, and sentence structure based on the provided toolcall feedback.

            2. **Logical Structure**:
            Evaluate clarity, coherence, and logical flow of arguments as indicated by the toolcall insights.

            3. **Information Value**:
            Determine the relevance, originality, and usefulness of the content, guided by toolcall indicators.

            4. **Readability**:
            Rate how easy, engaging, and audience-appropriate the content is, utilizing toolcall readability scores.

            5. **Content Safety**:
            Confirm the absence of harmful, offensive, or misleading content according to toolcall alerts and provide a safety rating.

            6. **Target Fit**:
            Judge alignment with audience expectations and needs as suggested by toolcall assessments.

            Make sure you provide your evaluations in the provided structured JSON format.

            Ensure your assessment is comprehensive, balanced,  and clearly justified.'"""}
    messages={'role': 'user',
            'content': content}
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "evaluate_response",
            "schema": {
                "type": "object",
                "properties": {
                    "dimension": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Rating": {
                                    "type": "number",
                                    "description": "from 0 to 10"
                                },
                                "Justification": {
                                    "type": "string",
                                    "description": "Brief summary referencing toolcall results"
                                },
                                "Suggestions for Improvement": {
                                    "type": "string",
                                    "description": "Provide actionable suggestions if rating is below 8; otherwise state 'None required.'"
                                }
                            },
                            "required": ["Rating", "Justification", "Suggestions for Improvement"],
                            "additionalProperties": False
                        }

                    },
                    "summary": {"type": "string"}
                },
                "required": ["dimension", "summary"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    conversations.append(prompt)
    conversations.append(messages)


    # print(conversations[-2:])
    response, result = models_list[0].generate_messages(conversations[-2:], response_format=response_format)

    new_content = {
        "role": "assistant",
        "content": result.get("content", "None")}
    conversations.append(new_content)
    return new_content['content']


@app.route('/')
def studio():
    return render_template('ai_studio_code.html',
                           langval = 0, logicval = 0,
                           infoval = 0,  readval = 0,
                           safetyval = 0, targetval = 0
                           )


def get_content(i):
    return contents[i]


def background_process(task_id):
    task_store[task_id] = {'status': 'done', 'result': main(get_content(task_id))}
    return task_store[task_id]


@app.route('/submit', methods=['POST'])
def submit():
    task_id = str(uuid.uuid4())
    if 'file' in request.files:
        content = request.files['file'].read().decode('utf-8')
    elif 'text' in request.form:
        content = request.form['text']
    else:
        return jsonify({'error': 'no content'}), 400

    task_store[task_id] = {'status': 'pending', 'result': None}
    contents[task_id] = content

    threading.Thread(target=background_process, args=(task_id,)).start()

    return jsonify({'task_id': task_id})


@app.route('/result')
def result():
    task_id = request.args.get('task_id')
    task = task_store.get(task_id)
    if not task:
        return jsonify({'status': 'not_found', 'result': None})
    if task["status"] != 'pending':
        dim_list = [
            "Language Correction",
            "Logical Structure",
            "Information Value",
            "Readability",
            "Content Safety",
            "Target Fit"
        ]
        resultlist = json.loads(task['result'])
        longtext = ""
        for i in range(6):
            longtext += dim_list[i] + ': ' + str(resultlist['dimension'][i]) + '\n'
        longtext += "summary: " + str(resultlist['summary']) + '\n'
        task['result'] = {
            "langval": resultlist['dimension'][0]["Rating"],
            "logicval": resultlist['dimension'][1]["Rating"],
            "infoval": resultlist['dimension'][2]["Rating"],
            "readval": resultlist['dimension'][3]["Rating"],
            "safetyval": resultlist['dimension'][4]["Rating"],
            "targetval": resultlist['dimension'][5]["Rating"],
            "output_text": longtext,
        }
    return jsonify(task)


if __name__ == "__main__":
    app.run(debug=True)
