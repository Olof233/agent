import utils
import models
import tools
from flask import Flask, request, render_template, jsonify
import uuid
import threading
import json
import io
import pdfplumber


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

            1. **语言正确性**:
            Assess accuracy in spelling, grammar, punctuation, and sentence structure based on the provided toolcall feedback.

            2. **逻辑结构**:
            Evaluate clarity, coherence, and logical flow of arguments as indicated by the toolcall insights.

            3. **信息价值**:
            Determine the relevance, originality, and usefulness of the content, guided by toolcall indicators.

            4. **可读性**:
            Rate how easy, engaging, and audience-appropriate the content is, utilizing toolcall readability scores.

            5. **合规安全**:
            Confirm the absence of harmful, offensive, or misleading content according to toolcall alerts and provide a safety rating.

            6. **目标契合度**:
            Judge alignment with audience expectations and needs as suggested by toolcall assessments.

            Make sure you provide your evaluations in the provided structured JSON format and response in Chinese.

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
                    "dimensions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "评分": {
                                    "type": "number",
                                    "description": "from 0 to 10"
                                },
                                "分析": {
                                    "type": "string",
                                    "description": "Brief summary referencing toolcall results"
                                },
                                "提升建议": {
                                    "type": "string",
                                    "description": "Provide actionable suggestions if rating is below 8; otherwise state 'Not required.'"
                                }
                            },
                            "required": ["评分", "分析", "提升建议"],
                            "additionalProperties": False
                        }

                    },
                    "summary": {"type": "string"}
                },
                "required": ["dimensions", "summary"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    conversations.append(prompt)
    conversations.append(messages)


    print(conversations[-2:])
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
        file = request.files['file']
        filetype = file.mimetype.split('/')[-1]
        if filetype == "txt" or "csv" or "json" or "md" or "log":
            content = file.read().decode('utf-8')
        elif filetype == "pdf":
            file_stream = io.BytesIO(file.read())
            content = ''
            with pdfplumber.open(file_stream) as pdf:
                for i in range(len(pdf.pages)):
                    page = pdf.pages[i]
                    text = page.extract_text()
                    if text:
                        page_content = '\n'.join(text.split('\n')[:-1])
                        content += page_content
        else:
            content = ''
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
            "语言正确性",
            "逻辑结构",
            "信息价值",
            "可读性",
            "合规安全",
            "目标契合度"
        ]
        resultlist = json.loads(task['result'])
        longtext = ""
        for i in range(6):
            longtext += dim_list[i] + ': ' + str(resultlist['dimensions'][i]) + '\n'
        longtext += "summary: " + str(resultlist['summary']) + '\n'
        task['result'] = {
            "langval": resultlist['dimensions'][0]["评分"],
            "logicval": resultlist['dimensions'][1]["评分"],
            "infoval": resultlist['dimensions'][2]["评分"],
            "readval": resultlist['dimensions'][3]["评分"],
            "safetyval": resultlist['dimensions'][4]["评分"],
            "targetval": resultlist['dimensions'][5]["评分"],
            "output_text": longtext,
        }
    return jsonify(task)


if __name__ == "__main__":
    app.run(debug=True)
