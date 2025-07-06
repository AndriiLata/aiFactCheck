import random
from testing import utils
import requests
from tqdm import tqdm
import pandas as pd
import os
from app.infrastructure.llm.llm_client import chat
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Script started")  # At the very top
API_URL = "https://verify-api-770851903956.europe-west3.run.app/api/verify"
MAX_SAMPLES = 1000  # Set your desired maximum here

# 1. Load and filter NEI samples
data = utils.load_fever_dataset("testing/Datasets/fever_dataset.jsonl", drop_NEI=False)
nei_samples = []
for claim, entry in data.items():
    label = entry["Label"][0]  # Label is a list, take first element
    evidence = entry["Evidence"]
    
    if label == "NOT ENOUGH INFO":
        nei_samples.append((claim, label, evidence))

print(f"Loaded {len(nei_samples)} NEI samples")

# Randomly select up to MAX_SAMPLES NEI claims
if len(nei_samples) > MAX_SAMPLES:
    nei_samples = random.sample(nei_samples, MAX_SAMPLES)

# 2. Classify NEI claims
results = []
for claim, true_label, evidence in tqdm(nei_samples, desc="Classifying NEI claims"):
    try:
        response = requests.post(API_URL, json={"claim": claim}, timeout=1000)
        if response.status_code == 200:
            raw_data = response.json()
            pred_label = raw_data.get("label", "NOT_ENOUGH_INFO")
            reason = raw_data.get("reason", "")
            
            # Handle different evidence field structures
            found_evidence = []
            found_evidence_formatted = ""  # For Excel display

            if "evidence" in raw_data and raw_data["evidence"]:
                found_evidence = raw_data["evidence"]
                print(f"Found web evidence - {len(found_evidence)} items")
                
                # Sort evidence by weight (highest first) - this is the most important ranking
                def get_weight(ev):
                    if isinstance(ev, dict):
                        return ev.get("weight", 0.0)
                    return 0.0
                
                # Sort by weight (highest first) and take top 10
                sorted_evidence = sorted(found_evidence, key=get_weight, reverse=True)
                top_evidence = sorted_evidence[:10]  # Take top 10 by weight
                
                if top_evidence:
                    best_weight = get_weight(top_evidence[0])
                    worst_weight = get_weight(top_evidence[-1]) if len(top_evidence) > 1 else best_weight
                
                # Format for Excel - create readable text
                evidence_parts = []
                for i, ev in enumerate(top_evidence, 1):
                    if isinstance(ev, dict):
                        snippet = ev.get("snippet", "")
                        source = ev.get("source", "")
                        trust = ev.get("trust", "")
                        confidence = ev.get("confidence", "")
                        weight = ev.get("weight", "")
                        nli = ev.get("nli", "")  # Add NLI score
                        nli_entailment = ev.get("entailment", "")  # Entailment score
                        nli_contradiction = ev.get("contradiction", "")  # Contradiction score
                        nli_neutral = ev.get("neutral", "")  # Neutral score
                        
                        evidence_text = f"[{i}] {snippet}"
                        if source:
                            evidence_text += f"\n    Source: {source}"
                        if trust:
                            evidence_text += f"\n    Trust: {trust:.3f}"
                        if confidence:
                            evidence_text += f"\n    Confidence: {confidence:.3f}"
                        if nli:
                            evidence_text += f"\n    NLI: {nli}"
                        
                        
                        evidence_parts.append(evidence_text)
                    else:
                        evidence_parts.append(f"[{i}] {str(ev)}")
                
                found_evidence_formatted = "\n\n".join(evidence_parts)
                if len(found_evidence) > 10:
                    found_evidence_formatted += f"\n\n... ({len(found_evidence) - 10} additional evidence items were found but not shown)"

            # Check if this is KG agent response (has "all_top_evidence_paths" field)
            elif "all_top_evidence_paths" in raw_data and raw_data["all_top_evidence_paths"]:
                
                # Format KG evidence for Excel
                kg_parts = []
                for i, path in enumerate(raw_data["all_top_evidence_paths"][:3], 1):  # Limit to top 3 paths
                    path_text = f"[Path {i}]"
                    for j, edge in enumerate(path, 1):
                        if isinstance(edge, dict):
                            subject = edge.get("subject", "").replace("http://dbpedia.org/resource/", "")
                            predicate = edge.get("predicate", "").replace("http://dbpedia.org/ontology/", "").replace("http://dbpedia.org/property/", "")
                            object_val = edge.get("object", "")
                            
                            # Clean up object value
                            if object_val.startswith("http://dbpedia.org/resource/"):
                                object_val = object_val.replace("http://dbpedia.org/resource/", "")
                            
                            edge_text = f"  {j}. {subject} → {predicate} → {object_val}"
                            path_text += f"\n{edge_text}"
                    
                    kg_parts.append(path_text)
                
                found_evidence_formatted = "\n\n".join(kg_parts)
                
                # Create KG evidence for processing
                kg_evidence = []
                for path in raw_data["all_top_evidence_paths"]:
                    for edge in path:
                        if isinstance(edge, dict):
                            subject = edge.get("subject", "")
                            predicate = edge.get("predicate", "")
                            object_val = edge.get("object", "")
                            
                            snippet = f"{subject} {predicate} {object_val}"
                            kg_evidence.append({
                                "snippet": snippet,
                                "source": "DBpedia",
                                "trust": 0.9,
                                "kg_edge": True
                            })
                found_evidence = kg_evidence

            else:
                print("No evidence found in either format")
                found_evidence = []
                found_evidence_formatted = "No evidence found"
                
        else:
            print(f"API Error {response.status_code} for claim: {claim[:50]}...")
            pred_label = "Error"
            found_evidence = []
            found_evidence_formatted = f"API Error: HTTP {response.status_code}"
            reason = f"HTTP {response.status_code}"
    except Exception as e:
        print(f"Exception for claim: {claim[:50]}... -> {e}")
        pred_label = "Error"
        found_evidence = []
        found_evidence_formatted = f"Exception: {str(e)}"
        reason = str(e)
        
        
    results.append({
        "claim": claim,
        "true_label": true_label,
        "predicted_label": pred_label,
        "found_evidence": found_evidence_formatted,  # Use formatted version for Excel
        "found_evidence_raw": found_evidence,        # Keep raw for LLM processing
        "reason": reason  
    })

print(f"Number of results: {len(results)}")


# 3. LLM explanation for misclassified NEI
def ask_llm_about_nei(claim, evidence):
    sys_prompt = (
        "You are analyzing fact-checking results. "
        "Does the evidence clearly support or refute the claim, or is it insufficient? "
        "If insufficient, explain why. If sufficient, explain why."
    )
    user_prompt = (
        f"Claim: {claim}\n"
        f"Evidence found: {evidence if evidence else 'No evidence found'}"
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        msg = chat(messages)
        return msg.content
    except Exception as e:
        return f"LLM Error: {str(e)}"

# Add LLM explanations for misclassified cases
print("Generating LLM explanations for misclassified cases...")
for entry in tqdm(results, desc="LLM explanations"):
    # Check if the prediction is NOT "Not Enough Info" 
    normalized_pred = entry["predicted_label"].lower().replace(" ", "").replace("_", "")
    if normalized_pred not in ["notenoughinfo", "notinfo"]:
        # Use raw evidence for LLM processing
        entry["llm_explanation"] = ask_llm_about_nei(entry["claim"], entry["found_evidence_raw"])
    else:
        entry["llm_explanation"] = None

# 4. Output
df = pd.DataFrame(results)

# Add row number as first column (starting from 1)
df.insert(0, 'nr', range(1, len(df) + 1))

# Add human annotation columns at the end
df['human_annotated'] = ''  # Empty for Excel checkboxes
df['notes'] = ''  # Optional notes column

base_filename = "nei_test_results.csv"
filename = base_filename
i = 1
while os.path.exists(filename):
    filename = f"nei_test_results_{i}.csv"
    i += 1

print("Saving results to CSV...")
print("Current working directory:", os.getcwd())

# Create a copy of df without unwanted columns for both CSV and Excel
columns_to_drop = ['reason', 'found_evidence_raw']
df_export = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Save CSV without unwanted columns
df_export.to_csv(filename, index=False, sep=';')
print(f"Results saved to {filename}")

# Create Excel version with better formatting
excel_filename = filename.replace('.csv', '.xlsx')

print("\nClassification Report:")
try:
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Use df_export (without unwanted columns) for Excel
        df_export.to_excel(writer, sheet_name='NEI_Results', index=False)
        
        # Get the worksheet to format
        worksheet = writer.sheets['NEI_Results']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
            worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Add header formatting
        from openpyxl.styles import Font, PatternFill
        header_font = Font(bold=True)
        header_fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
        
        for cell in worksheet[1]:  # First row (headers)
            cell.font = header_font
            cell.fill = header_fill
    
    print(f"Excel version saved to {excel_filename}")

    
except ImportError:
    print("openpyxl not installed. Install with: pip install openpyxl")
    print("Excel file not created, but CSV is available.")