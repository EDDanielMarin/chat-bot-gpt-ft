with open('data.json', 'r') as f:
    data = json.load(f)
fine_tune_data = []
for i in range(0, len(data['messages']), 2):
    prompt = data['messages'][i]['content']  
    completion = data['messages'][i+1]['content']  
    
    fine_tune_data.append({
        "prompt": prompt,
        "completion": completion
    })
with open('fine_tune_data.json', 'w') as f:
    json.dump(fine_tune_data, f, indent=2)