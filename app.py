from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class CurrentStudy(BaseModel):
    study_id: str
    study_description: str
    study_date: str

class PriorStudy(BaseModel):
    study_id: str
    study_description: str
    study_date: str

class Case(BaseModel):
    case_id: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    current_study: CurrentStudy
    prior_studies: List[PriorStudy]

class ChallengeRequest(BaseModel):
    challenge_id: Optional[str] = None
    schema_version: Optional[int] = None
    generated_at: Optional[str] = None
    cases: List[Case]

class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool

class ChallengeResponse(BaseModel):
    predictions: List[Prediction]

# ============================================================
# Body Part Keywords (ordered longest first)
# ============================================================
BODY_PART_KEYWORDS = [
    ("cervical spine", "cervical_spine"), ("c-spine", "cervical_spine"),
    ("cspine", "cervical_spine"), ("c spine", "cervical_spine"),
    ("thoracic spine", "thoracic_spine"), ("t-spine", "thoracic_spine"),
    ("tspine", "thoracic_spine"), ("t spine", "thoracic_spine"),
    ("lumbar spine", "lumbar_spine"), ("l-spine", "lumbar_spine"),
    ("lspine", "lumbar_spine"), ("l spine", "lumbar_spine"),
    ("sacroiliac", "lumbar_spine"), ("sacrum", "lumbar_spine"), ("coccyx", "lumbar_spine"),
    ("head/brain", "brain"), ("head brain", "brain"),
    ("brain", "brain"), ("cerebral", "brain"),
    ("eeg", "brain"), ("electroencephalog", "brain"),
    ("head", "head"), ("skull", "head"), ("cranial", "head"), ("intracranial", "brain"),
    ("sinus", "sinus"), ("orbit", "orbit"),
    ("facial", "face"), ("face", "face"), ("mandible", "face"), ("tmj", "face"),
    ("temporal", "temporal"), ("mastoid", "temporal"), ("iac", "iac"),
    ("carotid", "carotid"),
    ("soft tissue neck", "neck"), ("neck", "neck"), ("thyroid", "neck"),
    ("spine", "spine"),
    ("coronary", "heart"), ("cardiac", "heart"), ("heart", "heart"),
    ("echo", "heart"), ("tte", "heart"), ("transthorac", "heart"),
    ("transesophag", "heart"), ("aorta", "heart"), ("aortic", "heart"),
    ("ffr", "heart"), ("definity", "heart"),
    ("chest", "chest"), ("lung", "chest"), ("pulmonary", "chest"),
    ("pul ", "chest"), ("thorax", "chest"),
    ("ribs", "chest"), ("rib ", "chest"), ("sternum", "chest"), ("mediastin", "chest"),
    ("liver", "abdomen"), ("hepatic", "abdomen"), ("pancrea", "abdomen"),
    ("spleen", "abdomen"), ("gallbladder", "abdomen"), ("biliary", "abdomen"),
    ("mrcp", "abdomen"), ("adrenal", "abdomen"),
    ("renal", "kidney"), ("kidney", "kidney"),
    ("abdomen", "abdomen"), ("abdominal", "abdomen"), ("abd ", "abdomen"),
    ("abd/pel", "abdomen"), ("abd_pel", "abdomen"),
    ("abd/", "abdomen"),
    ("endovaginal", "pelvis"), ("transvaginal", "pelvis"),
    ("uterus", "pelvis"), ("uterine", "pelvis"),
    ("ovary", "pelvis"), ("ovarian", "pelvis"),
    ("prostate", "pelvis"), ("rectal", "pelvis"), ("rectum", "pelvis"),
    ("pelvis", "pelvis"), ("pelvic", "pelvis"),
    ("ob us", "pelvis"), ("ob ", "pelvis"),
    ("/pel", "pelvis"), ("_pel", "pelvis"),
    ("bladder", "bladder"),
    ("breast", "breast"), ("mammogram", "breast"), ("mammog", "breast"),
    ("mammo", "breast"), ("mam ", "breast"), ("mam-", "breast"),
    ("shoulder", "shoulder"), ("clavicle", "shoulder"), ("scapula", "shoulder"),
    ("humerus", "upper_arm"), ("elbow", "elbow"), ("forearm", "forearm"),
    ("wrist", "wrist"), ("hand", "hand"), ("finger", "hand"), ("thumb", "hand"),
    ("hip", "hip"),
    ("femur", "thigh"), ("thigh", "thigh"), ("knee", "knee"),
    ("tibia", "lower_leg"), ("fibula", "lower_leg"),
    ("lower extrem", "lower_extremity"), (" le ", "lower_extremity"),
    ("ankle", "ankle"), ("anklel", "ankle"),
    ("foot", "foot"), ("toe", "foot"), ("calcaneus", "foot"), ("heel", "foot"),
    ("angiogram", "vascular"), ("angiography", "vascular"),
    ("angio", "vascular"), ("mra", "vascular"), ("venous", "vascular"),
    ("whole body", "whole_body"), ("bone scan", "whole_body"),
    ("dxa", "bone_density"), ("dexa", "bone_density"), ("bone density", "bone_density"),
]

RELATED_PARTS = {
    ("brain", "head"), ("head", "brain"),
    ("wrist", "hand"), ("hand", "wrist"),
    ("abdomen", "pelvis"), ("pelvis", "abdomen"),
    ("kidney", "abdomen"), ("abdomen", "kidney"),
    ("pelvis", "bladder"), ("bladder", "pelvis"),
    ("abdomen", "bladder"), ("bladder", "abdomen"),
    ("lower_extremity", "knee"), ("lower_extremity", "ankle"),
    ("lower_extremity", "foot"), ("lower_extremity", "thigh"),
    ("lower_extremity", "lower_leg"),
    ("knee", "lower_extremity"), ("ankle", "lower_extremity"),
    ("foot", "lower_extremity"), ("thigh", "lower_extremity"),
    ("lower_leg", "lower_extremity"),
    ("spine", "cervical_spine"), ("spine", "thoracic_spine"), ("spine", "lumbar_spine"),
    ("cervical_spine", "spine"), ("thoracic_spine", "spine"), ("lumbar_spine", "spine"),
    ("ankle", "foot"), ("foot", "ankle"),
    ("cervical_spine", "neck"), ("neck", "cervical_spine"),
    # Heart/ECHO is relevant to chest
    ("heart", "chest"), ("chest", "heart"),
    # Carotid relates to brain (vascular supply)
    ("carotid", "brain"), ("brain", "carotid"),
    ("carotid", "head"), ("head", "carotid"),
    # Cervical spine and head/brain
    ("cervical_spine", "brain"), ("brain", "cervical_spine"),
    ("cervical_spine", "head"), ("head", "cervical_spine"),
}

LATERALITY_SENSITIVE = {
    "knee", "hip", "shoulder", "wrist", "hand", "elbow",
    "ankle", "foot", "breast", "upper_arm", "forearm",
    "lower_leg", "thigh", "orbit", "iac",
}

MODALITY_KEYWORDS = [
    ("mri ", "MRI"), ("mri-", "MRI"), ("mr ", "MRI"), ("magnetic", "MRI"),
    ("ct ", "CT"), ("ct-", "CT"), ("computed", "CT"),
    ("xr ", "XR"), ("xr-", "XR"), ("x-ray", "XR"), ("xray", "XR"), ("radiograph", "XR"),
    ("us ", "US"), ("ultrasound", "US"), ("sono", "US"),
    ("fluoro", "FLUORO"),
    ("mammo", "MAMMO"), ("mam ", "MAMMO"),
    ("pet", "PET"),
    ("nuclear", "NM"), ("nm ", "NM"), ("bone scan", "NM"),
    ("dxa", "DEXA"), ("dexa", "DEXA"), ("bone density", "DEXA"),
    ("echo", "ECHO"), ("tte", "ECHO"), ("transthorac", "ECHO"), ("transesophag", "ECHO"),
    ("angio", "ANGIO"), ("mra", "MRA"), ("cta", "CTA"),
    ("ir ", "IR"),
]

INCOMPATIBLE_MODALITIES = set()
for m in ["MRI", "CT", "US", "XR", "ECHO", "MAMMO", "NM", "ANGIO", "MRA", "CTA", "IR", "PET", "FLUORO"]:
    INCOMPATIBLE_MODALITIES.add(("DEXA", m))
    INCOMPATIBLE_MODALITIES.add((m, "DEXA"))


def extract_body_parts(description: str) -> set:
    desc_lower = " " + description.lower() + " "
    parts = set()
    for keyword, part in BODY_PART_KEYWORDS:
        if keyword in desc_lower:
            parts.add(part)
    return parts if parts else {"unknown"}


def extract_modality(description: str) -> str:
    desc_lower = " " + description.lower() + " "
    for keyword, modality in MODALITY_KEYWORDS:
        if keyword in desc_lower:
            return modality
    desc_upper = description.upper().strip()
    if desc_upper.startswith("CT"): return "CT"
    if desc_upper.startswith("MRI") or desc_upper.startswith("MR "): return "MRI"
    if desc_upper.startswith("XR"): return "XR"
    if desc_upper.startswith("US "): return "US"
    return "UNKNOWN"


def extract_laterality(description: str) -> str:
    desc = " " + description.upper() + " "
    if " BI " in desc or "BILATERAL" in desc or " BI-" in desc or " BI/" in desc:
        return "BILATERAL"
    desc_end = description.upper().strip()
    has_right = any(x in desc for x in [" RT ", " RT,", " RIGHT ", "RIGHT,"]) or desc_end.endswith(" RT") or desc_end.endswith(" RIGHT")
    has_left = any(x in desc for x in [" LT ", " LT,", " LEFT ", "LEFT,"]) or desc_end.endswith(" LT") or desc_end.endswith(" LEFT")
    if has_right and has_left: return "BILATERAL"
    if has_right: return "RIGHT"
    if has_left: return "LEFT"
    return "NONE"


def normalize_description(desc: str) -> str:
    desc = desc.upper().strip()
    desc = desc.replace("CNTRST", "CONTRAST").replace("CNTRS", "CONTRAST")
    desc = desc.replace("W/O", "WITHOUT").replace("WO ", "WITHOUT ")
    desc = desc.replace("W/", "WITH").replace("W CON", "WITH CONTRAST")
    desc = re.sub(r'\s+', ' ', desc)
    return desc


def are_parts_related(parts1: set, parts2: set) -> bool:
    if parts1 & parts2:
        return True
    for p1 in parts1:
        for p2 in parts2:
            if (p1, p2) in RELATED_PARTS:
                return True
    return False


def predict_relevance(current_study: CurrentStudy, prior_study: PriorStudy) -> bool:
    current_desc = current_study.study_description
    prior_desc = prior_study.study_description
    current_norm = normalize_description(current_desc)
    prior_norm = normalize_description(prior_desc)

    current_parts = extract_body_parts(current_desc)
    prior_parts = extract_body_parts(prior_desc)
    current_modality = extract_modality(current_desc)
    prior_modality = extract_modality(prior_desc)
    current_lat = extract_laterality(current_desc)
    prior_lat = extract_laterality(prior_desc)

    # RULE 0: DXA/bone density is only relevant to other DXA
    if (current_modality, prior_modality) in INCOMPATIBLE_MODALITIES:
        return False
    if "bone_density" in current_parts and "bone_density" not in prior_parts:
        return False
    if "bone_density" in prior_parts and "bone_density" not in current_parts:
        return False

    # RULE 1: Exact match
    if current_norm == prior_norm:
        return True

    # RULE 2: Check body part relevance
    parts_related = are_parts_related(current_parts, prior_parts)

    if not parts_related:
        # Fuzzy fallback for unknown parts
        if "unknown" in current_parts or "unknown" in prior_parts:
            current_words = set(current_norm.split())
            prior_words = set(prior_norm.split())
            stop_words = {"WITH", "WITHOUT", "AND", "OR", "THE", "OF", "CONTRAST",
                         "LIMITED", "COMPLETE", "BILATERAL", "RIGHT", "LEFT", "RT",
                         "LT", "UNILATERAL", "PA", "AP", "LATERAL", "PORTABLE",
                         "STAT", "VIEW", "VIEWS", "FRONTAL", "ONLY", "MIN",
                         "1V", "2V", "3V", "W", "WO", "CON", "CAD", "TOMO"}
            mod_words = {"MRI", "MR", "CT", "XR", "US", "ULTRASOUND", "PET",
                        "NM", "DEXA", "DXA", "ECHO", "MAM", "MAMMO", "IR", "FLUORO"}
            c_content = current_words - stop_words - mod_words
            p_content = prior_words - stop_words - mod_words
            if c_content and p_content and (c_content & p_content):
                return True
        return False

    # Parts ARE related — now apply filters

    # RULE 3: Laterality check
    shared_parts = current_parts & prior_parts
    # Check laterality for shared laterality-sensitive parts
    sensitive_shared = shared_parts & LATERALITY_SENSITIVE
    # Also check laterality for related laterality-sensitive parts
    all_current_sensitive = current_parts & LATERALITY_SENSITIVE
    all_prior_sensitive = prior_parts & LATERALITY_SENSITIVE
    has_sensitive = bool(sensitive_shared) or (bool(all_current_sensitive) and bool(all_prior_sensitive))
    
    if has_sensitive:
        if (current_lat in ("LEFT", "RIGHT") and
            prior_lat in ("LEFT", "RIGHT") and
            current_lat != prior_lat):
            return False

    # RULE 4: NM lymphoscint != mammogram/breast imaging
    dl_c = current_desc.lower()
    dl_p = prior_desc.lower()
    if "lymph" in dl_p and "lymph" not in dl_c:
        return False
    if "lymph" in dl_c and "lymph" not in dl_p:
        return False

    # RULE 5: Biopsy/procedure != diagnostic imaging (except breast)
    bx_keywords = ["stereo bx", "biopsy", "bx ", "needle loc", "aspiration"]
    prior_is_bx = any(k in dl_p for k in bx_keywords)
    current_is_bx = any(k in dl_c for k in bx_keywords)
    # Breast biopsies ARE relevant to breast imaging
    both_breast = "breast" in current_parts and "breast" in prior_parts
    if not both_breast:
        if prior_is_bx and not current_is_bx:
            return False
        if current_is_bx and not prior_is_bx:
            return False

    # RULE 6: "PORTABLE" XR views (like abdomen portable) are just imaging
    # This is fine, no special handling needed

    # Parts related, laterality OK, no exclusion rules triggered -> relevant
    return True


@app.get("/")
def health_check():
    return {"status": "ok", "challenge": "relevant-priors-v1"}


@app.post("/")
def predict(request: ChallengeRequest) -> ChallengeResponse:
    logger.info(f"Received request with {len(request.cases)} cases")
    predictions = []
    for case in request.cases:
        for prior in case.prior_studies:
            is_relevant = predict_relevance(case.current_study, prior)
            predictions.append(Prediction(
                case_id=case.case_id,
                study_id=prior.study_id,
                predicted_is_relevant=is_relevant
            ))
    relevant_count = sum(1 for p in predictions if p.predicted_is_relevant)
    logger.info(f"Processed {len(request.cases)} cases, {sum(len(c.prior_studies) for c in request.cases)} priors, {relevant_count} relevant")
    return ChallengeResponse(predictions=predictions)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
