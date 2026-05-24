#%%
import pandas as pd
import re
import os
#%%
# 1. Environment Setup
print("Loading annotation and ED mapping data...")
annotation_path = 'datasets/toxcast_annotation/assay_annotations_invitrodb_v4_3_AUG2024.xlsx'
ed_mapping_path = 'datasets/toxcast_annotation/ED_result.xlsx' 
toxcast_input_path = 'datasets/raw_data/chem_dataset/toxcast.csv'
output_path = 'datasets/processed/toxcast/processed_toxcast.csv'
# %%
# Helper function: Name normalization and direction extraction
def clean_assay_name(n): return re.sub(r"_(Positive|Negative|dn|up|down)$", "", re.sub(r"(\d+)h(?!r)", r"\1hr", str(n)), flags=re.IGNORECASE)
#%%
def extract_direction(n):
    if re.search(r"_(dn|down|antagonist|inhibitor|Negative)", str(n), re.I): return "Inhibition/Down-regulation"
    if re.search(r"_(up|agonist|Positive)", str(n), re.I): return "Activation/Up-regulation"
    return "Neutral/Standard"
#%%
# 2. Build ED result mapping dictionary (AEID -> {MoA, Tier, Target Gene})
def build_ed_mapping(file_path):
    df = pd.read_excel(file_path, header=None)
    tier_row = df.iloc[0]
    moa_row = df.iloc[1]
    aeid_row = df.iloc[2]
    gene_row = df.iloc[4]
    
    mapping = {}
    current_tier = "unknown"
    current_moa = "unknown"
    
    for i in range(14, len(aeid_row)):
        if pd.notna(tier_row[i]): current_tier = str(tier_row[i]).strip()
        if pd.notna(moa_row[i]): current_moa = str(moa_row[i]).strip()
        
        if pd.notna(aeid_row[i]):
            try:
                aid_int = int(float(aeid_row[i]))
                mapping[aid_int] = {
                    "ed_tier": current_tier,
                    "ed_moa": current_moa,
                    "target_gene": str(gene_row[i]) if pd.notna(gene_row[i]) else "unknown"
                }
            except: continue
    return mapping

ed_mapping = build_ed_mapping(ed_mapping_path)
print(f"Mapped {len(ed_mapping)} AEIDs from ED result.")
#%%
def get_context(assay, lookup_dict, ed_mapping, context_columns):
    # [Core] Convert all search keys to lowercase for comparison
    assay_lower = str(assay).lower()
    
    # 1. Full name priority mapping (lowercase basis)
    res = lookup_dict.get(assay_lower)
    
    if not res:
        # 2. Cleaned Name mapping (lowercase basis)
        cleaned = clean_assay_name(assay).lower()
        res = lookup_dict.get(cleaned)
    
    if not res:
        # 3. Prefix mapping (last resort, lowercase basis)
        prefix = "_".join(assay_lower.split('_')[:2])
        for k, v in lookup_dict.items():
            if k.startswith(prefix):
                res = v
                break
    
    final_res = res.copy() if res else {c: "Unknown" for c in context_columns}
    
    # Extract directionality (use original name)
    final_res['directionality'] = extract_direction(assay)
    
    # [ED Result Mapping] aeid is numeric, so case has no effect
    aeid_val = final_res.get('aeid')
    if aeid_val and aeid_val != "Unknown":
        try:
            aid_int = int(float(aeid_val))
            if aid_int in ed_mapping:
                final_res['ed_tier'] = ed_mapping[aid_int]['ed_tier']
                final_res['ed_moa'] = ed_mapping[aid_int]['ed_moa']
                final_res['target_gene'] = ed_mapping[aid_int]['target_gene']
            else:
                final_res['ed_moa'] = final_res['ed_tier'] = final_res['target_gene'] = "unknown"
        except:
            final_res['ed_moa'] = final_res['ed_tier'] = final_res['target_gene'] = "unknown"
    else:
        final_res['ed_moa'] = final_res['ed_tier'] = final_res['target_gene'] = "unknown"
        
    return final_res
#%%
def main():
    print("Building mappings...")
    ed_mapping = build_ed_mapping(ed_mapping_path)
    
    context_columns = [
        'aeid', 'assay_source_name', 'cell_format', 'timepoint_hr', 
        'biological_process_target', 'intended_target_type', 
        'intended_target_family', 'organism', 'cell_viability_assay', 'assay_function_type'
    ]
    
    annotations = pd.read_excel(annotation_path, sheet_name='annotations_combined')
    lookup_dict = {}
    for _, row in annotations.iterrows():
        for col in ['assay_component_endpoint_name', 'assay_component_name', 'assay_name']:
            name = str(row[col])
            if pd.notna(name):
                # [Core] Always save keys in lowercase when building the dictionary
                name_key = name.lower()
                if name_key not in lookup_dict:
                    lookup_dict[name_key] = {c: row[c] if pd.notna(row[c]) else "Unknown" for c in context_columns}
    print("Processing ToxCast raw data...")
    reader = pd.read_csv(toxcast_input_path, chunksize=100000)
    processed_chunks = []
    total_rows = 0
    drop_prefixes = 'ACEA_T47D_80hr_|APR_Hepat_|NCCT_|NHEERL_'
    
    for i, chunk in enumerate(reader):
        total_rows += len(chunk)
        # [Core Fix] Resolve argument passing issue using lambda
        chunk['context'] = chunk['assay'].apply(
            lambda x: get_context(x, lookup_dict, ed_mapping, context_columns)
        )
        chunk = chunk[~chunk['assay'].str.contains(drop_prefixes, na=False)]
        processed_chunks.append(chunk)
        print(f"OK: {total_rows} rows processed...")

    df_final = pd.concat(processed_chunks)
    print(f"Final Data Shape: {df_final.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Data saved successfully to {output_path}")
#%%
if __name__ == "__main__":
    main()
# %%