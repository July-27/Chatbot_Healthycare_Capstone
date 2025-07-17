import os
import xml.etree.ElementTree as ET
import json

# System prompt cố định
SYSTEM_PROMPT = (
    "You are a professional medical assistant, providing accurate and understandable information about medical conditions."
)

def extract_qa_pairs(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    qa_pairs = root.find('QAPairs')
    pairs = []
    if qa_pairs is not None:
        for qa in qa_pairs.findall('QAPair'):
            question = qa.find('Question').text.strip().replace('\n', ' ')
            answer = qa.find('Answer').text.strip().replace('\n', ' ')
            # Loại bỏ khoảng trắng thừa
            question = ' '.join(question.split())
            answer = ' '.join(answer.split())
            pairs.append((question, answer))
    return pairs

def format_llama2_conversation(qa_pairs, system_prompt=SYSTEM_PROMPT):
    # Định dạng hội thoại với system prompt ở đầu
    sys = f"<<SYS>>{system_prompt}</SYS>>"
    segments = []
    for idx, (q, a) in enumerate(qa_pairs):
        if idx == 0:
            segments.append(f"<s>[INST] {sys}{q} [/INST] {a} </s>")
        else:
            segments.append(f"<s>[INST] {q} [/INST] {a} </s>")
    return ''.join(segments)

def convert_folder_to_jsonl(input_folder, output_jsonl):
    with open(output_jsonl, 'w', encoding='utf-8') as out_f:
        for filename in os.listdir(input_folder):
            if filename.endswith('.xml'):
                xml_path = os.path.join(input_folder, filename)
                qa_pairs = extract_qa_pairs(xml_path)
                if qa_pairs:
                    text = format_llama2_conversation(qa_pairs)
                    obj = {"text": text}
                    out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')

# Ví dụ sử dụng:
convert_folder_to_jsonl('1_CancerGov_QA', '1_CancerGov_QA.jsonl')