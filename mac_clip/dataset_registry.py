"""
Dataset Registry
================
Maps each dataset key to its modality-specific clinical concept prompts.
The main model calls ``model.set_dataset_concepts(dataset_name)`` before each
training run to hot-swap the prompts without re-instantiating the model.
"""

from typing import Dict, Any

# ---------------------------------------------------------------------------
# Per-dataset clinical concept prompts
# ---------------------------------------------------------------------------
DATASET_CONCEPTS: Dict[str, Dict[str, Any]] = {
    "kvasir": {
        "clinical_concepts": [
            "irregular border lesion colonoscopy",
            "smooth rounded lesion colonoscopy",
            "granular mucosal surface endoscopy",
            "flat sessile lesion mucosa",
            "erythematous lesion colon mucosa",
            "pale lesion colon background",
            "small polyp focal lesion",
            "large polyp broad base",
        ],
        "boundary_concept": "boundary edge lesion margin colonoscopy",
        "interior_concept": "interior region polyp tissue colonoscopy",
    },
    "isic2018": {
        "clinical_concepts": [
            "melanoma dermoscopy irregular border",
            "benign nevus dermoscopy smooth border",
            "pigmented lesion dermoscopy dark",
            "seborrheic keratosis dermoscopy",
            "basal cell carcinoma dermoscopy",
            "actinic keratosis skin lesion",
            "vascular lesion dermoscopy red",
            "skin lesion dermoscopy background",
        ],
        "boundary_concept": "boundary edge skin lesion margin dermoscopy",
        "interior_concept": "interior region skin lesion tissue dermoscopy",
    },
    "m2caiseg": {
        "clinical_concepts": [
            "surgical instrument endoscopy tool",
            "laparoscopic forceps abdominal",
            "laparoscopic grasper tissue manipulation",
            "clip applier laparoscopic surgery",
            "scissors laparoscopic instrument",
            "hook electrocautery laparoscopy",
            "irrigator laparoscopic tool",
            "laparoscopic instrument background tissue",
        ],
        "boundary_concept": "boundary edge surgical instrument laparoscopy",
        "interior_concept": "interior region surgical instrument metal",
    },
    "bkai_neopolyp": {
        "clinical_concepts": [
            "neoplastic polyp colon colonoscopy",
            "non-neoplastic polyp colonoscopy",
            "sessile serrated lesion colon",
            "pedunculated polyp stalk",
            "flat polyp subtle elevation",
            "hyperplastic polyp pale mucosa",
            "adenoma colon lesion",
            "polyp colon background mucosa",
        ],
        "boundary_concept": "boundary edge polyp margin colon",
        "interior_concept": "interior region polyp tissue colon",
    },
    "covid_infection": {
        "clinical_concepts": [
            "COVID-19 lung opacity chest X-ray",
            "pneumonia consolidation chest radiograph",
            "ground glass opacity lung infection",
            "bilateral infiltrates chest X-ray",
            "pleural effusion chest radiograph",
            "normal lung chest X-ray",
            "non-COVID pneumonia infiltrate",
            "lung infection region radiograph",
        ],
        "boundary_concept": "boundary edge lung infection opacity chest X-ray",
        "interior_concept": "interior region lung infection consolidation",
    },
    "covid_lung": {
        "clinical_concepts": [
            "lung parenchyma chest X-ray",
            "lung field chest radiograph",
            "right lung lobe chest X-ray",
            "left lung lobe chest radiograph",
            "lung boundary diaphragm ribs",
            "lung hyperinflation emphysema",
            "atelectasis lung collapse",
            "normal lung parenchyma chest",
        ],
        "boundary_concept": "boundary edge lung lobe chest X-ray diaphragm",
        "interior_concept": "interior region lung parenchyma chest radiograph",
    },
    "monuseg": {
        "clinical_concepts": [
            "cell nucleus histology hematoxylin eosin",
            "nuclei pathology tissue slide",
            "tumor cell nucleus cancer pathology",
            "inflammatory cell nucleus tissue",
            "epithelial cell nucleus gland",
            "stromal cell nucleus connective tissue",
            "mitotic figure cell division",
            "nucleus background cytoplasm tissue",
        ],
        "boundary_concept": "boundary edge cell nucleus membrane histology",
        "interior_concept": "interior region cell nucleus chromatin histology",
    },
    "busi": {
        "clinical_concepts": [
            "breast lesion ultrasound hypoechoic",
            "benign breast mass ultrasound oval",
            "malignant breast mass ultrasound irregular",
            "breast cyst ultrasound anechoic",
            "fibroadenoma breast ultrasound",
            "breast carcinoma ultrasound spiculated",
            "breast mass posterior enhancement",
            "breast tissue background ultrasound",
        ],
        "boundary_concept": "boundary edge breast lesion ultrasound margin",
        "interior_concept": "interior region breast lesion tissue ultrasound",
    },
    "ct2us_kidney": {
        "clinical_concepts": [
            "kidney ultrasound renal cortex",
            "renal parenchyma ultrasound echogenicity",
            "kidney border ultrasound capsule",
            "renal medulla ultrasound hypoechoic",
            "kidney cyst ultrasound anechoic",
            "renal mass ultrasound lesion",
            "normal kidney ultrasound shape",
            "kidney background abdominal ultrasound",
        ],
        "boundary_concept": "boundary edge kidney capsule ultrasound margin",
        "interior_concept": "interior region kidney parenchyma ultrasound",
    },
    "arcade_stenosis": {
        "clinical_concepts": [
            "coronary artery stenosis angiography",
            "coronary vessel narrowing X-ray",
            "coronary artery lesion angiogram",
            "coronary plaque obstruction vessel",
            "coronary bifurcation stenosis",
            "left main coronary stenosis",
            "right coronary artery stenosis",
            "coronary artery background angiography",
        ],
        "boundary_concept": "boundary edge coronary stenosis vessel angiography",
        "interior_concept": "interior region coronary artery lumen angiography",
    },
    "arcade_syntax": {
        "clinical_concepts": [
            "coronary artery vessel angiography",
            "coronary artery tree anatomy",
            "coronary vessel segment angiogram",
            "left anterior descending artery",
            "circumflex coronary artery",
            "right coronary artery angiography",
            "coronary branch vessel",
            "coronary artery background cardiac",
        ],
        "boundary_concept": "boundary edge coronary artery vessel wall angiography",
        "interior_concept": "interior region coronary artery lumen vessel",
    },
    "btxrd": {
        "clinical_concepts": [
            "bone tumor X-ray lesion",
            "osteosarcoma bone radiograph",
            "bone metastasis lesion X-ray",
            "pathological fracture bone tumor",
            "bone cyst X-ray lytic",
            "benign bone tumor X-ray",
            "cortical bone destruction radiograph",
            "bone tumor background normal bone",
        ],
        "boundary_concept": "boundary edge bone tumor margin X-ray",
        "interior_concept": "interior region bone tumor tissue radiograph",
    },
    "papila": {
        "clinical_concepts": [
            "optic disc fundus glaucoma",
            "optic cup fundus retina",
            "optic nerve head fundus photography",
            "cup to disc ratio fundus",
            "retinal nerve fiber layer fundus",
            "glaucomatous optic disc cup",
            "normal optic disc fundus",
            "optic disc background retina",
        ],
        "boundary_concept": "boundary edge optic disc cup fundus margin",
        "interior_concept": "interior region optic disc cup fundus retina",
    },
    "drive": {
        "clinical_concepts": [
            "retinal vessel fundus photography",
            "retinal artery fundus blood vessel",
            "retinal vein fundus vessel",
            "retinal vessel bifurcation fundus",
            "small retinal capillary fundus",
            "large retinal vessel fundus",
            "retinal vessel macula fundus",
            "retinal background fundus photography",
        ],
        "boundary_concept": "boundary edge retinal vessel wall fundus",
        "interior_concept": "interior region retinal vessel lumen fundus",
    },
    "coph100": {
        "clinical_concepts": [
            "retinopathy of prematurity OCT fundus",
            "neonatal retina vascularization",
            "retinal ridge zone vascularization",
            "avascular retina neonatal fundus",
            "neovascularization retina premature",
            "retinal vessel development neonatal",
            "stage ROP retina fundus",
            "retina background neonatal fundus",
        ],
        "boundary_concept": "boundary edge retinal lesion ROP fundus margin",
        "interior_concept": "interior region retinal vascular zone fundus",
    },
}

# ---------------------------------------------------------------------------
# Modality grouping (used for radar charts and per-modality tables)
# ---------------------------------------------------------------------------
MODALITY_MAP: Dict[str, str] = {
    "kvasir":          "Endoscopy",
    "bkai_neopolyp":   "Endoscopy",
    "m2caiseg":        "Endoscopy",
    "covid_infection": "CXR",
    "covid_lung":      "CXR",
    "arcade_stenosis": "X-Ray",
    "arcade_syntax":   "X-Ray",
    "btxrd":           "X-Ray",
    "isic2018":        "Dermoscopy",
    "monuseg":         "Pathology",
    "busi":            "Ultrasound",
    "ct2us_kidney":    "Ultrasound",
    "papila":          "Fundus/OCT",
    "drive":           "Fundus/OCT",
    "coph100":         "Fundus/OCT",
}

# Datasets excluded from aggregate means (systematic overfitting in val pool)
EXCLUDED_FROM_AGGREGATES = {"coph100"}
