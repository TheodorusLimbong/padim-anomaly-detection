import docx
import re

doc_path = r"D:\skripsi\self supervised\draft skripsi\Proposal_Theodorus Limbong_223020503095_revisi.docx"
doc = docx.Document(doc_path)

in_bab4 = False
bab4_paragraphs = []

for i, para in enumerate(doc.paragraphs):
    text = para.text.strip()
    if not text:
        continue
    upper = text.upper()
    if "BAB IV" in upper and ("HASIL" in upper or "PEMBAHASAN" in upper):
        in_bab4 = True
        print(f"BAB IV starts at para {i}")
        continue
    if in_bab4:
        if ("BAB V" in upper) or (upper.startswith("BAB V") or ("KESIMPULAN" in upper and "SARAN" in upper)):
            print(f"BAB V starts at para {i}")
            in_bab4 = False
            continue
        bab4_paragraphs.append((i, text))

print(f"Total BAB IV paragraphs: {len(bab4_paragraphs)}")

# Identify section headers
section_headers = {}
for idx, (i, text) in enumerate(bab4_paragraphs):
    m = re.match(r'^\s*(\d+\.\d+(?:\.\d+)?)\s', text)
    if m:
        section_num = m.group(1)
        section_headers[section_num] = {"start_idx": idx, "text": text[:120], "para_num": i}

print("\n=== SECTION STRUCTURE ===")
for sec in sorted(section_headers.keys()):
    info = section_headers[sec]
    print(f"  {sec}: para_idx={info['start_idx']}, doc_para={info['para_num']}: {info['text']}")

# Assign sections
para_sections = []
current_section = "intro"
for idx, (i, text) in enumerate(bab4_paragraphs):
    m = re.match(r'^\s*(\d+\.\d+(?:\.\d+)?)\s', text)
    if m:
        current_section = m.group(1)
    para_sections.append(current_section)

# Define analytical keywords
analytical_kw = [
    r'\b(karena|sebab|sehingga|maka|berarti|dengan demikian|hal ini menunjuk)',
    r'\b(menunjuk|mengindikasi|menggambar|mengungkap|mengkonfirmasi)',
    r'\b(unggul|lebih baik|lebih tinggi|lebih rendah|signifikan|superior|kompetitif)',
    r'\b(analisis|analisa|interpretasi|dapat dilihat|terlihat|teramati)',
    r'\b(disebabkan|dipengaruh|berkaitan|berhubungan|korelasi)',
    r'\b(perbandingan|dibanding|perbeda)',
    r'\b(meskipun|walaupun|namun|sedangkan|sementara)',
]

def has_analytical(text):
    for pat in analytical_kw:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

def find_metrics(text):
    found = []
    if re.search(r'\bAUROC\b', text, re.IGNORECASE):
        found.append("AUROC")
    if re.search(r'(Pixel|pixel|PIXEL).{0,10}(AUROC|AUC)', text, re.IGNORECASE):
        found.append("Pixel AUROC")
    if re.search(r'(PRO|pro|Pro)(\s*|\-)(score|Score|SCORE)', text, re.IGNORECASE):
        found.append("PRO-score")
    if re.search(r'\b(Precision|precision|presisi)\b', text, re.IGNORECASE):
        found.append("Precision")
    if re.search(r'\b(Recall|recall)\b', text, re.IGNORECASE):
        found.append("Recall")
    if re.search(r'\b(F1|f1)\b', text, re.IGNORECASE):
        found.append("F1-score")
    if re.search(r'\b(Threshold|threshold|ambang batas|P\d{2})\b', text, re.IGNORECASE):
        found.append("Threshold")
    if re.search(r'\b(Accuracy|accuracy|akurasi)\b', text, re.IGNORECASE):
        found.append("Accuracy")
    if re.search(r'\b(mean|rata(-|\s*)rata|rerata|std|standar deviasi)\b', text, re.IGNORECASE):
        found.append("Mean/Std")
    if re.search(r'\b(gap|selisih)\b', text, re.IGNORECASE):
        found.append("Gap")
    if re.search(r'\b(inference time|lama inferensi|waktu inferensi|detik|sekon)\b', text, re.IGNORECASE):
        found.append("Inference Time")
    if re.search(r'\b(1792|dimensionality|dimensi)\b', text, re.IGNORECASE):
        found.append("Dimensionality")
    if re.search(r'\b(Gaussian|Mahalanobis|covariance|kovarian)\b', text, re.IGNORECASE):
        found.append("Statistical")
    if re.search(r'\b(skor|score)\b', text, re.IGNORECASE):
        found.append("Score reference")
    if re.search(r'\b\d+[.,]\d+%\b', text):
        found.append("Percentage")
    if re.search(r'\btrue positive|false positive|TPR|FPR\b', text, re.IGNORECASE):
        found.append("TPR/FPR")
    return found

print("\n\n========== DETAILED ANALYSIS ==========")
results = []
for idx, (i, text) in enumerate(bab4_paragraphs):
    sec = para_sections[idx]
    
    metrics = find_metrics(text)
    numbers = re.findall(r'\b\d+[.,]\d+\b', text)
    analytical = has_analytical(text)
    
    is_metric_para = bool(metrics) or (len(numbers) >= 2 and any(0 < float(n.replace(',','.')) <= 2 for n in numbers[:5]))
    is_table_like = ('\t' in text or text.count('  ') > 5 or text.startswith('|'))
    
    if is_metric_para and not is_table_like:
        if analytical:
            verdict = "ANALYTICAL"
        else:
            verdict = "DESCRIPTIVE (numbers only, no why)"
        
        results.append({
            "idx": idx,
            "doc_para": i,
            "section": sec,
            "metrics": metrics,
            "values": numbers[:5],
            "verdict": verdict,
            "text_preview": text[:200].replace('\n', ' '),
            "text_full": text
        })
        print(f"\n[{i}] Sec {sec} | {verdict}")
        print(f"  Metrics: {metrics}")
        print(f"  Values: {numbers[:5]}")
        print(f"  {text[:200].replace(chr(10), ' ')}")

print(f"\n\n=== TOTAL METRIC PARAGRAPHS: {len(results)} ===")

# Summary by section
print("\n\n=== SUMMARY BY SECTION ===")
by_section = {}
for r in results:
    sec = r["section"]
    if sec not in by_section:
        by_section[sec] = {"analytical": 0, "descriptive": 0, "total": 0}
    by_section[sec]["total"] += 1
    if r["verdict"] == "ANALYTICAL":
        by_section[sec]["analytical"] += 1
    else:
        by_section[sec]["descriptive"] += 1

for sec in sorted(by_section.keys()):
    d = by_section[sec]
    print(f"  Section {sec}: {d['total']} paragraphs ({d['analytical']} analytical, {d['descriptive']} descriptive/number-only)")

# Save full results
with open(r"D:\skripsi\self supervised\code\metrics_analysis_results.txt", "w", encoding="utf-8") as f:
    f.write("=== METRIC PARAGRAPH ANALYSIS - BAB IV ===\n\n")
    for r in results:
        f.write(f"\n===== Para {r['doc_para']} | Section {r['section']} | {r['verdict']} =====\n")
        f.write(f"Metrics: {', '.join(r['metrics']) if r['metrics'] else 'none'}\n")
        f.write(f"Values: {', '.join(r['values']) if r['values'] else 'none'}\n")
        f.write(f"Full text:\n{r['text_full']}\n")
        f.write("=" * 80 + "\n")
    
    f.write(f"\n\n=== SUMMARY ===\n")
    f.write(f"Total metric paragraphs: {len(results)}\n")
    for sec in sorted(by_section.keys()):
        d = by_section[sec]
        f.write(f"Section {sec}: {d['total']} paragraphs ({d['analytical']} analytical, {d['descriptive']} descriptive)\n")

print("\nFull results saved to metrics_analysis_results.txt")
