"""
scripts/generate_synthetic_data.py
Generates all 35 synthetic insurance documents and saves them as .txt files
under data/synthetic/<category>/.

Document counts:
  policies/          → 4 files
  provider_agreements/ → 3 files
  historical_claims/  → 25 files  (one JSON-like .txt per claim)
  regulations/        → 3 files
"""
import sys
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from faker import Faker

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_SYNTHETIC_DIR
from loguru import logger

fake = Faker("en_IN")
random.seed(42)
fake.seed_instance(42)

# ── Helpers ───────────────────────────────────────────────────────────────────
def save(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info(f"  Wrote {path.relative_to(DATA_SYNTHETIC_DIR.parent.parent)}")


def rand_inr(lo: int, hi: int) -> str:
    return f"₹{random.randint(lo, hi):,}"


def rand_date(years_back: int = 3) -> str:
    d = fake.date_between(start_date=f"-{years_back}y", end_date="today")
    return d.strftime("%d %B %Y")


# ── ICD-10 / Procedure pool ───────────────────────────────────────────────────
DIAGNOSES = [
    ("A90",   "Dengue fever"),
    ("I21.0", "Acute myocardial infarction — anterior wall"),
    ("M17.11","Primary osteoarthritis, right knee"),
    ("E11.9", "Type 2 diabetes mellitus without complications"),
    ("N18.5", "Chronic kidney disease, stage 5"),
    ("C34.1", "Malignant neoplasm of upper lobe of bronchus"),
    ("K35.2", "Acute appendicitis with generalised peritonitis"),
    ("H26.9", "Cataract, unspecified"),
    ("I63.5", "Cerebral infarction due to occlusion of cerebral arteries"),
    ("J18.9", "Pneumonia, unspecified organism"),
]

PROCEDURES = [
    ("27447", "Total knee arthroplasty"),
    ("33533", "Coronary artery bypass, arterial"),
    ("44950", "Appendectomy"),
    ("66984", "Cataract extraction with IOL"),
    ("90935", "Hemodialysis procedure"),
    ("99213", "Office/outpatient visit, established patient"),
    ("31622", "Bronchoscopy, diagnostic"),
    ("43239", "Upper GI endoscopy with biopsy"),
    ("70553", "MRI brain with contrast"),
    ("93306", "Echocardiography, complete"),
]

INSURERS = [
    "Star Health Insurance",
    "HDFC ERGO Health",
    "New India Assurance",
    "ICICI Lombard Health",
]

HOSPITALS = [
    "Apollo Hospitals",
    "Fortis Healthcare",
    "Manipal Hospitals",
    "Max Healthcare",
]

TPAS = ["Medi Assist", "Paramount Health", "Vidal Health", "Heritage Health"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INSURANCE POLICIES (4 documents)
# ═══════════════════════════════════════════════════════════════════════════════
POLICY_TYPES = [
    ("individual", "Individual Health Insurance Plan", "POL-2024-IND"),
    ("group",      "Group Health Insurance Plan",      "POL-2024-GRP"),
    ("corporate",  "Corporate Floater Health Plan",    "POL-2024-CORP"),
    ("senior",     "Senior Citizen Health Plan",       "POL-2024-SEN"),
]

POLICY_TEMPLATE = """\
INSURANCE POLICY DOCUMENT
=========================
Policy Name:    {policy_name}
Policy Number:  {policy_number}-001
Insurer:        {insurer}
Effective Date: {eff_date}
Expiry Date:    {exp_date}
Premium:        {premium} per annum
Sum Insured:    {sum_insured}

─────────────────────────────────────────────────────────────────
SECTION 1 — DEFINITIONS
─────────────────────────────────────────────────────────────────
1.1  "Insured Person" means the policyholder and any dependants
     listed in the schedule of this policy.
1.2  "Network Hospital" means a hospital that has a cashless
     arrangement with the TPA empanelled by the insurer.
1.3  "Pre-existing Disease (PED)" means any condition, ailment,
     injury or disease that was diagnosed or for which medical
     advice or treatment was recommended or received within
     48 months prior to the first policy issuance.
1.4  "Day Care Treatment" means medical treatment or surgical
     procedure requiring less than 24 hours of hospitalisation.
1.5  "TPA" means Third Party Administrator licensed by IRDA.

─────────────────────────────────────────────────────────────────
SECTION 2 — SCOPE OF COVERAGE
─────────────────────────────────────────────────────────────────
2.1  In-patient Hospitalisation:
     The insurer shall reimburse or arrange cashless payment for
     in-patient hospitalisation expenses incurred within India,
     subject to the sum insured of {sum_insured}.

2.2  Pre-hospitalisation Expenses:
     Covered up to 30 days prior to hospitalisation.

2.3  Post-hospitalisation Expenses:
     Covered up to 60 days after discharge.

2.4  Day Care Procedures:
     {num_daycare} day care procedures listed in Annexure A are
     covered even if hospitalisation is less than 24 hours.

2.5  Organ Donor Expenses:
     Medical expenses of the organ donor are covered up to
     {organ_limit} per policy year.

─────────────────────────────────────────────────────────────────
SECTION 3 — SURGICAL PROCEDURES & SUB-LIMITS
─────────────────────────────────────────────────────────────────
3.1  General Surgery:
     Covered up to the sum insured with no sub-limit.

3.2  Joint Replacement (Knee / Hip):
     Sub-limit: {joint_limit} per joint, maximum 2 joints
     per policy year. Pre-authorisation mandatory.
     Waiting period: 24 months from policy inception.

3.3  Cardiac Procedures (CABG, Angioplasty):
     Sub-limit: {cardiac_limit}. Pre-authorisation required.
     Minimum hospitalisation: 48 hours.

3.4  Cataract Surgery:
     Sub-limit: {cataract_limit} per eye.
     Day care eligible; no overnight stay required.

3.5  Dialysis (Hemodialysis / Peritoneal):
     Covered per session up to {dialysis_limit} per session,
     maximum 156 sessions per policy year.
     Applicable only after 90-day waiting period from inception.

3.6  Cancer Treatment:
     Covered including chemotherapy, radiation, and surgery.
     Sub-limit: {cancer_limit} per policy year.

─────────────────────────────────────────────────────────────────
SECTION 4 — EXCLUSIONS
─────────────────────────────────────────────────────────────────
4.1  Standard Exclusions (permanent):
     a) Cosmetic or aesthetic treatment
     b) Infertility and assisted reproduction
     c) Self-inflicted injury or attempted suicide
     d) War, invasion, acts of foreign enemy
     e) Experimental treatment not approved by medical authorities

4.2  Pre-existing Condition Waiting Period:
     Pre-existing diseases are excluded for the first 48 months
     of the policy. After 48 months of continuous coverage,
     pre-existing conditions are covered subject to:
     - Sub-limit: 80% of sum insured for first claim
     - Required documentation: discharge summary + specialist
       letter confirming PED diagnosis date

4.3  Specific Disease Waiting Periods (from policy inception):
     - Joint replacement:          24 months
     - Cataract:                   24 months
     - Hernia, Hydrocele:          12 months
     - Benign ENT disorders:       12 months
     - Kidney stones:              12 months
     - Diabetes-related complications: 12 months (Type 2)

4.4  First 30-Day Exclusion:
     No claims are payable for any illness arising within the
     first 30 days of the initial policy, except accidents.

─────────────────────────────────────────────────────────────────
SECTION 5 — CLAIM PROCEDURE
─────────────────────────────────────────────────────────────────
5.1  Cashless Claims (Network Hospitals):
     a) Inform the TPA at least 48 hours before planned admission
        (4 hours for emergencies).
     b) Present the health card at the hospital insurance desk.
     c) The TPA will issue a pre-authorisation letter within
        2 hours for emergencies and 6 hours for planned cases.
     d) Final discharge approval: within 4 hours of discharge.

5.2  Reimbursement Claims (Non-network Hospitals):
     a) Notify TPA within 24 hours of admission.
     b) Collect all original bills, discharge summary, pharmacy
        receipts, investigation reports.
     c) Submit claim form within 30 days of discharge.
     d) Reimbursement will be processed within 30 days of
        receiving complete documentation.

5.3  Required Documents for All Claims:
     - Filled and signed claim form
     - Original hospital discharge summary
     - Original bills and payment receipts
     - Attending physician's certificate
     - Investigation reports (lab, imaging)
     - Pre-authorisation letter (if cashless)
     - NEFT details for reimbursement

─────────────────────────────────────────────────────────────────
SECTION 6 — CO-PAYMENT & DEDUCTIBLES
─────────────────────────────────────────────────────────────────
6.1  Standard Co-payment: {copay}% of each claim amount.
6.2  Senior Citizen Co-payment: An additional 10% applies for
     insured persons aged 60 years and above.
6.3  Non-network Hospital: Additional 20% co-payment applies.
6.4  Deductible: {deductible} per policy year (applies before
     the insurer's liability commences).

─────────────────────────────────────────────────────────────────
SECTION 7 — RENEWAL & PORTABILITY
─────────────────────────────────────────────────────────────────
7.1  This policy is renewable annually subject to the insurer's
     board-approved underwriting guidelines.
7.2  No-Claim Bonus (NCB): 10% increase in sum insured for each
     claim-free year, up to a maximum of 50%.
7.3  Portability rights under IRDA (Health Insurance) Regulations
     2016 are available. Waiting periods already served are
     credited to the new policy.

─────────────────────────────────────────────────────────────────
SECTION 8 — GRIEVANCE REDRESSAL
─────────────────────────────────────────────────────────────────
8.1  Contact the insurer's grievance cell within 15 days of the
     decision. Resolution within 15 working days.
8.2  Unresolved complaints may be escalated to the Insurance
     Ombudsman as per IRDA Circular IRDA/MISC/CIR/GRV/054/2011.

─────────────────────────────────────────────────────────────────
SECTION 9 — EXPERIMENTAL TREATMENT EXCLUSION
─────────────────────────────────────────────────────────────────
9.1  Treatments not approved by the Drug Controller General of
     India (DCGI) or equivalent international authority are
     excluded. This includes gene therapy, stem cell therapy
     (unless for haematological malignancies with established
     protocol), and unproven biological therapies.
"""


def generate_policies() -> None:
    for ptype, pname, pnum in POLICY_TYPES:
        insurer = random.choice(INSURERS)
        eff_date = fake.date_between(start_date="-2y", end_date="-1y")
        exp_date = eff_date + timedelta(days=365)
        content = POLICY_TEMPLATE.format(
            policy_name   = pname,
            policy_number = pnum,
            insurer       = insurer,
            eff_date      = eff_date.strftime("%d %B %Y"),
            exp_date      = exp_date.strftime("%d %B %Y"),
            premium       = rand_inr(8_000, 35_000),
            sum_insured   = rand_inr(3_00_000, 25_00_000),
            num_daycare   = random.randint(140, 180),
            organ_limit   = rand_inr(50_000, 1_50_000),
            joint_limit   = rand_inr(1_00_000, 2_50_000),
            cardiac_limit = rand_inr(3_00_000, 6_00_000),
            cataract_limit= rand_inr(25_000, 60_000),
            dialysis_limit= rand_inr(1_200, 3_500),
            cancer_limit  = rand_inr(5_00_000, 10_00_000),
            copay         = random.choice([0, 10, 20]),
            deductible    = rand_inr(0, 10_000),
        )
        fname = f"{ptype}_health_policy_{insurer.split()[0].lower()}.txt"
        save(DATA_SYNTHETIC_DIR / "policies" / fname, content)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PROVIDER AGREEMENTS (3 documents)
# ═══════════════════════════════════════════════════════════════════════════════
AGREEMENT_TEMPLATE = """\
HOSPITAL-TPA PROVIDER AGREEMENT
================================
Agreement Number: AGR-2024-{num:03d}
Hospital:         {hospital}
TPA:              {tpa}
Effective Date:   {eff_date}
Expiry Date:      {exp_date}

─────────────────────────────────────────────────────────────────
SECTION 1 — SCOPE
─────────────────────────────────────────────────────────────────
This agreement governs the cashless hospitalisation services
provided by {hospital} to insured members of the health insurance
products administered by {tpa}.

─────────────────────────────────────────────────────────────────
SECTION 2 — NEGOTIATED TARIFF RATES
─────────────────────────────────────────────────────────────────
2.1  General Ward (per day):          {gen_ward}
2.2  Semi-private Room (per day):     {semi_pvt}
2.3  Private Room (per day):          {pvt_room}
2.4  ICU / ICCU (per day):            {icu}
2.5  Operation Theatre (per hour):    {ot_per_hr}

Procedure-Specific Package Rates:
2.6  Total Knee Arthroplasty (TKA):  {tka_rate}
2.7  CABG (open heart):               {cabg_rate}
2.8  Cataract (phaco + IOL):          {cataract_rate}
2.9  Laparoscopic Appendectomy:       {appendix_rate}
2.10 Haemodialysis (per session):     {dialysis_rate}

─────────────────────────────────────────────────────────────────
SECTION 3 — CASHLESS AUTHORISATION
─────────────────────────────────────────────────────────────────
3.1  The hospital will submit pre-authorisation requests via
     the TPA portal within 2 hours of patient admission.
3.2  The TPA will respond within 2 hours for emergencies and
     6 hours for elective procedures.
3.3  Final bill submission within 24 hours of discharge.
3.4  Payment by TPA within 21 working days of complete
     documentation submission.

─────────────────────────────────────────────────────────────────
SECTION 4 — DISCOUNT STRUCTURE
─────────────────────────────────────────────────────────────────
4.1  The hospital agrees to a {discount}% discount on MRP for
     all drugs and consumables supplied during treatment.
4.2  Implants supplied at cost + {implant_markup}% margin.
4.3  Investigation charges at a flat {investigation_discount}%
     discount on published laboratory/radiology tariff.
"""


def generate_agreements() -> None:
    pairs = [
        (HOSPITALS[0], TPAS[0]),
        (HOSPITALS[1], TPAS[1]),
        (HOSPITALS[2], TPAS[2]),
    ]
    for i, (hospital, tpa) in enumerate(pairs, start=1):
        eff_date = fake.date_between(start_date="-2y", end_date="-6M")
        exp_date = eff_date + timedelta(days=365)
        content = AGREEMENT_TEMPLATE.format(
            num                   = i,
            hospital              = hospital,
            tpa                   = tpa,
            eff_date              = eff_date.strftime("%d %B %Y"),
            exp_date              = exp_date.strftime("%d %B %Y"),
            gen_ward              = rand_inr(1_200, 2_500),
            semi_pvt              = rand_inr(2_500, 5_000),
            pvt_room              = rand_inr(5_000, 10_000),
            icu                   = rand_inr(8_000, 20_000),
            ot_per_hr             = rand_inr(5_000, 15_000),
            tka_rate              = rand_inr(1_20_000, 2_20_000),
            cabg_rate             = rand_inr(2_50_000, 5_00_000),
            cataract_rate         = rand_inr(18_000, 45_000),
            appendix_rate         = rand_inr(35_000, 75_000),
            dialysis_rate         = rand_inr(1_000, 2_500),
            discount              = random.randint(5, 20),
            implant_markup        = random.randint(5, 15),
            investigation_discount= random.randint(10, 25),
        )
        fname = f"agreement_{hospital.split()[0].lower()}_{tpa.split()[0].lower()}.txt"
        save(DATA_SYNTHETIC_DIR / "provider_agreements" / fname, content)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HISTORICAL CLAIMS (25 documents)
# ═══════════════════════════════════════════════════════════════════════════════
OUTCOMES = ["approved", "rejected", "partially_approved"]
OUTCOME_WEIGHTS = [0.55, 0.25, 0.20]

REJECTION_REASONS = [
    "Pre-existing condition — 48-month waiting period not completed",
    "Specific disease waiting period of 24 months not satisfied",
    "Procedure falls under permanent exclusion (cosmetic)",
    "Sub-limit exceeded for joint replacement",
    "Claim submitted beyond 30-day deadline",
    "Incomplete documentation — discharge summary missing",
    "Non-network hospital — reimbursement rate applied, balance not covered",
    "First 30 days exclusion — illness arose within 30 days of policy start",
]


def generate_claims() -> None:
    for i in range(1, 26):
        dx_code, dx_desc = random.choice(DIAGNOSES)
        proc_code, proc_desc = random.choice(PROCEDURES)
        outcome = random.choices(OUTCOMES, weights=OUTCOME_WEIGHTS)[0]
        claimed = random.randint(30_000, 5_00_000)
        if outcome == "approved":
            approved = claimed
            reason = None
        elif outcome == "rejected":
            approved = 0
            reason = random.choice(REJECTION_REASONS)
        else:
            approved = int(claimed * random.uniform(0.4, 0.85))
            reason = "Sub-limit applied — partial reimbursement"

        claim_date = fake.date_between(start_date="-2y", end_date="-30d")
        decision_date = claim_date + timedelta(days=random.randint(3, 21))
        age = random.randint(25, 72)
        policy_type = random.choice(["IND", "GRP", "CORP", "SEN"])

        content = f"""\
HISTORICAL CLAIM RECORD
=======================
Claim ID:         CLM-2024-{i:05d}
Patient Name:     {fake.name()}
Age:              {age}
Policy Number:    POL-2024-{policy_type}-001
Insurer:          {random.choice(INSURERS)}
Hospital:         {random.choice(HOSPITALS)}
TPA:              {random.choice(TPAS)}

Diagnosis Code:   {dx_code}
Diagnosis:        {dx_desc}
Procedure Code:   {proc_code}
Procedure:        {proc_desc}

Claim Date:       {claim_date.strftime('%d %B %Y')}
Decision Date:    {decision_date.strftime('%d %B %Y')}
Claimed Amount:   ₹{claimed:,}
Approved Amount:  ₹{approved:,}
Outcome:          {outcome.upper().replace('_', ' ')}
{f"Rejection Reason: {reason}" if reason else ""}

Decision Notes:
The claim was reviewed by the medical team at {random.choice(TPAS)}.
{"Coverage confirmed under the applicable policy clause. All documentation verified." if outcome == "approved" else ""}
{"The claim could not be approved due to the following reason: " + (reason or "") if outcome == "rejected" else ""}
{"Partial reimbursement approved. " + (reason or "") if outcome == "partially_approved" else ""}
The patient's case was evaluated against policy terms and conditions
effective on the date of admission. The decision is final subject to
the grievance redressal process outlined in Section 8 of the policy.
"""
        fname = f"claim_CLM_2024_{i:05d}.txt"
        save(DATA_SYNTHETIC_DIR / "historical_claims" / fname, content)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. IRDA REGULATIONS (3 documents)
# ═══════════════════════════════════════════════════════════════════════════════
REGULATION_DOCS = [
    {
        "title": "IRDA Health Insurance Regulations 2016",
        "ref":   "IRDA/HLT/REG/CIR/2016/001",
        "fname": "irda_health_insurance_regulations_2016.txt",
        "body": """\
IRDA HEALTH INSURANCE REGULATIONS 2016
Circular Reference: IRDA/HLT/REG/CIR/2016/001
Issued by: Insurance Regulatory and Development Authority of India

─────────────────────────────────────────────────────────────────
REGULATION 1 — PRE-AUTHORISATION TIMELINES
─────────────────────────────────────────────────────────────────
1.1  For planned (elective) hospitalisation, the TPA must issue
     a pre-authorisation decision within 6 hours of receiving
     a complete request from the network hospital.
1.2  For emergency hospitalisation, the TPA must issue an initial
     authorisation within 1 hour and a final authorisation within
     6 hours of admission.
1.3  Failure to respond within the stipulated time shall be
     deemed as automatic approval for the requested amount,
     subject to policy limits.

─────────────────────────────────────────────────────────────────
REGULATION 2 — CASHLESS CLAIM DENIAL REQUIREMENTS
─────────────────────────────────────────────────────────────────
2.1  Any denial of cashless claims must be communicated in writing
     with specific reason citing the applicable policy clause.
2.2  Generic denial reasons ("not covered under policy") are
     not acceptable. The specific exclusion or sub-limit must
     be cited with the clause number.
2.3  The insured has the right to submit additional documents
     within 7 days of denial to challenge the decision.

─────────────────────────────────────────────────────────────────
REGULATION 3 — PORTABILITY RIGHTS
─────────────────────────────────────────────────────────────────
3.1  Any insured person may port their health insurance policy
     to another insurer at the time of renewal.
3.2  Waiting periods already served under the previous policy
     shall be credited to the new policy.
3.3  The new insurer cannot deny portability on grounds of
     pre-existing diseases that were previously covered.
3.4  Portability request must be submitted at least 45 days
     before the renewal date.

─────────────────────────────────────────────────────────────────
REGULATION 4 — STANDARDISATION OF EXCLUSIONS
─────────────────────────────────────────────────────────────────
4.1  All health insurance policies must follow the standard
     exclusion list issued by IRDA. Insurers may not add
     exclusions beyond this list without prior IRDA approval.
4.2  The maximum waiting period for any pre-existing disease
     shall not exceed 48 months.
4.3  The maximum waiting period for specific diseases shall
     not exceed 24 months from the date of policy inception.
""",
    },
    {
        "title": "IRDA Circular on Claim Settlement Timelines 2020",
        "ref":   "IRDA/HLT/CIR/MISC/2020/089",
        "fname": "irda_claim_settlement_timelines_2020.txt",
        "body": """\
IRDA CIRCULAR — CLAIM SETTLEMENT TIMELINES
Circular Reference: IRDA/HLT/CIR/MISC/2020/089
Issued by: Insurance Regulatory and Development Authority of India

─────────────────────────────────────────────────────────────────
CIRCULAR 1 — MANDATORY TIMELINES
─────────────────────────────────────────────────────────────────
1.1  All health insurance claims must be settled within 30 days
     of receiving the last necessary document.
1.2  If investigation is required, the insurer must inform the
     claimant within 30 days of filing, and complete the
     investigation within 45 days.
1.3  Delay beyond stipulated periods attracts interest at 2%
     per annum above the bank rate.

─────────────────────────────────────────────────────────────────
CIRCULAR 2 — CASHLESS DISCHARGE AUTHORISATION
─────────────────────────────────────────────────────────────────
2.1  Final discharge authorisation must be issued within
     4 hours of the hospital submitting the final bill.
2.2  The TPA cannot withhold discharge authorisation for
     reasons unrelated to the current claim.
2.3  Any queries on the final bill must be raised within
     2 hours of receiving the bill.

─────────────────────────────────────────────────────────────────
CIRCULAR 3 — GRIEVANCE REDRESSAL
─────────────────────────────────────────────────────────────────
3.1  All insurers must maintain a 24×7 toll-free helpline.
3.2  Written grievances must receive an acknowledgement within
     3 working days and resolution within 15 working days.
3.3  Unresolved grievances may be escalated to the Insurance
     Ombudsman within 1 year of the insurer's final response.
""",
    },
    {
        "title": "IRDA Guidelines on Standardised Health Insurance Products 2019",
        "ref":   "IRDA/HLT/REG/CIR/2019/101",
        "fname": "irda_standardised_health_products_2019.txt",
        "body": """\
IRDA GUIDELINES — STANDARDISED HEALTH INSURANCE PRODUCTS
Circular Reference: IRDA/HLT/REG/CIR/2019/101
Issued by: Insurance Regulatory and Development Authority of India

─────────────────────────────────────────────────────────────────
GUIDELINE 1 — AROGYA SANJEEVANI POLICY
─────────────────────────────────────────────────────────────────
1.1  All insurers must offer the IRDA-mandated Arogya Sanjeevani
     Policy with standard terms across the industry.
1.2  Sum insured options: ₹1 lakh to ₹5 lakh.
1.3  Co-payment: 5% of each claim.
1.4  Waiting periods: 30 days (initial), 48 months (PED),
     24 months (specific diseases).
1.5  No room rent sub-limit shall apply.

─────────────────────────────────────────────────────────────────
GUIDELINE 2 — MENTAL HEALTH COVERAGE
─────────────────────────────────────────────────────────────────
2.1  In compliance with the Mental Healthcare Act 2017,
     all health insurance policies issued or renewed after
     1 October 2022 must cover mental illness hospitalisation
     at par with physical illness.
2.2  Outpatient psychiatric treatment up to ₹5,000 per annum
     must be included in base coverage.

─────────────────────────────────────────────────────────────────
GUIDELINE 3 — COVID-19 COVERAGE
─────────────────────────────────────────────────────────────────
3.1  All standard health insurance policies must cover
     hospitalisation due to COVID-19 and its complications.
3.2  Home care treatment for COVID-19 is covered for a
     maximum of 14 days if prescribed by a qualified physician.

─────────────────────────────────────────────────────────────────
GUIDELINE 4 — TELEMEDICINE AND DIGITAL HEALTH
─────────────────────────────────────────────────────────────────
4.1  Telemedicine consultations as defined under the Telemedicine
     Practice Guidelines 2020 are covered under OPD benefit.
4.2  Insurers must accept digital health records (DigiLocker,
     ABHA records) as valid documentation for claims.
""",
    },
]


def generate_regulations() -> None:
    for doc in REGULATION_DOCS:
        content = f"DOCUMENT TITLE: {doc['title']}\nREFERENCE: {doc['ref']}\n\n{doc['body']}"
        save(DATA_SYNTHETIC_DIR / "regulations" / doc["fname"], content)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    logger.info("=" * 60)
    logger.info("  Generating synthetic insurance documents …")
    logger.info("=" * 60)

    generate_policies()
    generate_agreements()
    generate_claims()
    generate_regulations()

    # Count files
    total = sum(1 for _ in DATA_SYNTHETIC_DIR.rglob("*.txt"))
    logger.success(f"\n✓ Generated {total} synthetic documents in {DATA_SYNTHETIC_DIR}")
    logger.info("Run scripts/setup_mongo.py then scripts/ingest_documents.py next.")


if __name__ == "__main__":
    main()
