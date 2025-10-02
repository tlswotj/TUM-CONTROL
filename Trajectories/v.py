import json

# 원본 파일 열기
with open('reftraj_monteblanco_edgar.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# ref_v 항목의 값에 0.8을 곱함
data['ref_v'] = [v * 0.5 for v in data['ref_v']]

# 수정된 데이터를 새 파일로 저장
with open('reftraj_monteblanco_edgar_modified.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("ref_v 모든 값을 0.8로 수정하여 저장했습니다.")
