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
model = utils.load_models_from_config(config)[0]
# tool_definitions = tools.init_tools()

def evaluate(content):
    conversations = []
    prompt={'role': 'system',
            'content': utils.prompt.eva_prompt}
    messages={'role': 'user',
            'content': content}
    conversations.append(prompt)
    conversations.append(messages)


    print(conversations[-2:])
    response, result = model.generate_messages(conversations[-2:], response_format=utils.format.eva_format)

    new_content = {
        "role": "assistant",
        "content": result.get("content", "None")}
    conversations.append(new_content)
    return new_content['content']

def generate(content):
    conversations = []
    prompt={'role': 'system',
            'content': utils.prompt.gen_prompt}
    messages={'role': 'user',
            'content': content}
    conversations.append(prompt)
    conversations.append(messages)


    print(conversations[-2:])
    response, result = model.generate_messages(conversations[-2:], response_format=utils.format.gen_format)

    new_content = {
        "role": "assistant",
        "content": result.get("content", "None")}
    conversations.append(new_content)
    return new_content['content']


def get_content(i):
    return contents[i]


def background_process(task_id, mode):
    func = evaluate if mode == "eva" else (generate if mode == "gen" else lambda x: None)
    task_store[task_id] = {'status': 'done', 'result': func(get_content(task_id))}
    return task_store[task_id]



@app.route('/', methods=['GET'])
def studio():
    return render_template('ai_studio_code.html')


@app.route('/eva', methods=['GET'])
def eva():
    return render_template('ai_studio_eva.html',
                           langval = 0, logicval = 0,
                           infoval = 0,  readval = 0,
                           safetyval = 0, targetval = 0
                           )


@app.route('/gen', methods=['GET'])
def gen():
    return render_template('ai_studio_gen.html')


@app.route('/submit', methods=['POST'])
def submit():
    task_id = str(uuid.uuid4())
    print(request.files)
    if 'file' in request.files:
        file = request.files['file']
        filetype = file.mimetype.split('/')[-1]
        if filetype in ["txt", "csv", "json", "md", "log"]:
            content = file.read().decode('utf-8')
        elif filetype == "pdf":
            file_stream = io.BytesIO(file.read())
            content = ''
            with pdfplumber.open(file_stream) as pdf:
                for page in pdf.pages:
                    tables = page.find_tables()
                    table_bboxes = [table.bbox for table in tables]
                    
                    texts = page.extract_words()  # 每个词都有 bbox
                    
                    for word in texts:
                        word_bbox = word['x0'], word['top'], word['x1'], word['bottom']
                        # 判断是否在某个表格的 bbox 中
                        in_table = any(
                            word_bbox[0] >= bbox[0] and word_bbox[2] <= bbox[2] and
                            word_bbox[1] >= bbox[1] and word_bbox[3] <= bbox[3]
                            for bbox in table_bboxes
                        )
                        if not in_table:
                            content += word['text']
        else:
            content = ''
    elif 'text' in request.form:
        content = request.form['text']
    else:
        return jsonify({'error': 'no content'}), 400

    task_store[task_id] = {'status': 'pending', 'result': None}
    contents[task_id] = content

    threading.Thread(target=background_process, args=(task_id, request.form["mode"])).start()

    return jsonify({'task_id': task_id})


@app.route('/eva_result', methods=['GET'])
def eva_result():
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
        longtext += "总结: " + str(resultlist['summary']) + '\n'
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


@app.route('/gen_result', methods=['GET'])
def gen_result():
    task_id = request.args.get('task_id')
    task = task_store.get(task_id)
    if not task:
        return jsonify({'status': 'not_found', 'result': None})
    if task["status"] != 'pending':
        dim_list = [
            "目标",
            "核心方案",
            "执行步骤",
            "资源需求",
            "风险与应对",
            "预期效果"
        ]
        resultdict = json.loads(task['result'])
        longtext = ""
        dim = 0
        for i in resultdict['dimensions'].keys():
            longtext += dim_list[dim] + ': ' + str(resultdict['dimensions'][i].replace('\n', '')) + '\n'
            dim += 1
        longtext += "总结: " + str(resultdict['summary']) + '\n'
        task['result'] = {
            "output_text": longtext,
        }
    return jsonify(task)



if __name__ == "__main__":
    app.run(debug=True)
