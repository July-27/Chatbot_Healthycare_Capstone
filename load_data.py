import os
import xml.etree.ElementTree as ET
import json

def parse_medquad_file(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    source = root.attrib.get("source", "")
    url = root.attrib.get("url", "")
    focus = root.findtext("Focus", default="")

    qapairs = []
    for qapair in root.findall(".//QAPair"):
        question = qapair.findtext("Question", default="").strip()
        answer = qapair.findtext("Answer", default="").strip()
        qid = qapair.find("Question").attrib.get("qid", "")

        if question and answer:
            qapairs.append({
                "question": question,
                "answer": answer,
                "source": source,
                "focus": focus,
                "qid": qid,
                "url": url
            })

    return qapairs

def parse_medquad_directory(base_dir):
    all_qapairs = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root, file)
                qapairs = parse_medquad_file(xml_path)
                all_qapairs.extend(qapairs)
    return all_qapairs

if __name__ == "__main__":
    base_dir = "data/MedQuAD/MedQuAD"  # Đường dẫn gốc chứa các thư mục con như 1_CancerGov_QA
    output_path = "medquad_dataset.jsonl"

    all_data = parse_medquad_directory(base_dir)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Đã lưu {len(all_data)} QA pairs vào {output_path}")
