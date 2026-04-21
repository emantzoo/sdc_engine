"""
SDC Configuration and Constants
================================

Centralized configuration for all SDC methods, detection rules, and thresholds.

This module contains:
- Method implementation quality matrix
- QI detection keywords and scores
- Context-based protection thresholds
- Rule registry for method selection
- Default parameters and tuning schedules
"""

from typing import Dict, List, Any

# =============================================================================
# METHOD IMPLEMENTATION QUALITY MATRIX
# =============================================================================

METHOD_IMPLEMENTATION_QUALITY: Dict[str, Dict[str, Any]] = {
    'kANON': {
        'python_quality': 'OPTIMAL',
        'r_quality': 'OPTIMAL',
        'quality_gap': 'NONE',
        'r_package': 'sdcMicro',
        'default_to_r': False,
        'description': 'k-Anonymity via generalization',
        'data_type': 'microdata',
    },
    'PRAM': {
        'python_quality': 'OPTIMAL',
        'r_quality': 'OPTIMAL',
        'quality_gap': 'NONE',
        'r_package': 'sdcMicro',
        'default_to_r': False,
        'description': 'Post-RAndomization Method',
        'data_type': 'microdata',
    },
    'NOISE': {
        'python_quality': 'OPTIMAL',
        'r_quality': 'OPTIMAL',
        'quality_gap': 'NONE',
        'r_package': 'sdcMicro',
        'default_to_r': False,
        'description': 'Random noise addition',
        'data_type': 'microdata',
    },
    'LOCSUPR': {
        'python_quality': 'OPTIMAL',
        'r_quality': 'OPTIMAL',
        'quality_gap': 'NONE',
        'r_package': 'sdcMicro',
        'default_to_r': False,
        'description': 'Local suppression for k-anonymity',
        'data_type': 'microdata',
    },
}

# Method categories — microdata only (tabular methods removed)
MICRODATA_METHODS = ['kANON', 'PRAM', 'NOISE', 'LOCSUPR']
TABULAR_METHODS: List[str] = []

# =============================================================================
# DIRECT IDENTIFIER KEYWORDS (must be EXCLUDED before SDC)
# =============================================================================

DIRECT_IDENTIFIER_KEYWORDS: Dict[str, List[str]] = {
    # Personal names
    'name': ['name', 'firstname', 'first_name', 'lastname', 'last_name',
             'surname', 'fullname', 'full_name', 'middlename', 'middle_name'],

    # Contact information
    'email': ['email', 'mail', 'e_mail', 'email_address'],
    'phone': ['phone', 'telephone', 'mobile', 'cell', 'fax', 'tel'],
    'address': ['address', 'street', 'addr', 'residence',
                'home_address', 'work_address', 'postal_address'],

    # Government/National IDs
    'national_id': ['ssn', 'social_security', 'nin', 'national_id', 'national_insurance',
                    'passport', 'passport_no', 'driver_license', 'drivers_license',
                    'license_no', 'tax_id', 'tin', 'vat', 'afm', 'amka'],

    # Financial identifiers
    'financial': ['credit_card', 'card_number', 'account_number', 'iban',
                  'bank_account', 'cvv', 'pin'],

    # Securities & entity identifiers (financial/trading domains)
    'securities': ['lei', 'isin', 'cusip', 'sedol', 'cif', 'bic', 'swift'],

    # Temporal identifiers (high-precision timestamps that identify events)
    'temporal': ['timestamp', 'datetime'],

    # Medical identifiers
    'medical_id': ['medical_record', 'health_id', 'patient_id', 'mrn',
                   'medical_record_number', 'health_record'],

    # Generic ID columns (any column with 'id' as a word, e.g. HH_ID, record_id)
    'other_identifier': ['id', 'uuid', 'guid', 'record_id', 'row_id', 'key',
                         'index_id', 'unique_id', 'identifier'],
}

# Regex patterns for value-based detection
DIRECT_IDENTIFIER_PATTERNS: Dict[str, str] = {
    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    # Phone: require +prefix, parens, or 10+ pure digits to avoid matching
    # numeric ranges like "1967-1975" or "14848-30000"
    'phone': r'^(?:\+[0-9]{1,4}[\s-]?|[(][0-9]{1,4}[)][\s-]?)[0-9][-\s\./0-9]{6,}$|^[0-9]{10,}$',
    'credit_card': r'^[0-9]{13,19}$',
    'ssn_us': r'^\d{3}-\d{2}-\d{4}$',
    'iban': r'^[A-Z]{2}[0-9]{2}[A-Z0-9]{4,}$',
}

# =============================================================================
# QI DETECTION KEYWORDS AND SCORES
# =============================================================================

QI_KEYWORDS: Dict[str, Any] = {
    # Direct identifiers - always exclude from QIs (reference to above)
    'direct_identifiers': [
        'id', 'ssn', 'passport', 'email', 'phone', 'name', 'address',
        'license', 'account', 'card', 'social_security', 'employee_id',
        'patient_id', 'customer_id', 'user_id', 'member_id',
    ],

    # Definite QIs - high confidence scores
    'definite_qis': {
        'age': 1.00,
        'gender': 1.00,
        'sex': 1.00,
        'zipcode': 1.00,
        'zip': 1.00,
        'postal': 1.00,
        'postcode': 1.00,
        'race': 1.00,
        'ethnicity': 1.00,
        'ethnic': 1.00,
        'occupation': 0.95,
        'job': 0.90,
        'profession': 0.95,
        'city': 0.95,
        'town': 0.90,
        'county': 0.95,
        'state': 0.90,
        'province': 0.90,
        'region': 0.90,
        'country': 0.85,
        'nationality': 0.90,
        'marital': 0.95,
        'education': 0.95,
        'birthdate': 1.00,
        'dob': 1.00,
        'birth_date': 1.00,
        'birth_year': 0.95,
        'birth_month': 0.90,
    },

    # Probable QIs - medium confidence scores
    'probable_qis': {
        'year': 0.70,
        'month': 0.65,
        'date': 0.60,
        'income': 0.75,
        'salary': 0.75,
        'wage': 0.70,
        'religion': 0.80,
        'language': 0.70,
        'height': 0.65,
        'weight': 0.65,
        'diagnosis': 0.80,
        'icd': 0.75,
        'department': 0.60,
        'unit': 0.55,
        'ward': 0.60,
        'specialty': 0.60,
        'employer': 0.70,
        'company': 0.65,
        'organization': 0.60,
    },

    # Possible QIs - lower confidence scores
    'possible_qis': {
        'type': 0.40,
        'category': 0.45,
        'group': 0.40,
        'class': 0.45,
        'level': 0.40,
        'status': 0.50,
        'code': 0.35,
        'flag': 0.30,
    },

    # QI override patterns - columns that should be QIs despite matching identifier patterns
    'qi_override_patterns': [
        'zip', 'postal', 'diagnosis', 'icd', 'cpt', 'drg',
    ],
}

# Greek QI keywords — parallel to QI_KEYWORDS for Greek column names.
# Both accented and unaccented forms to handle encoding variations.
QI_KEYWORDS_GR: Dict[str, Any] = {
    # Greek direct identifiers — exclude from QIs
    'direct_identifiers_gr': [
        'αφμ', 'afm', 'αμκα', 'amka', 'αδτ', 'adt',
        'ονοματεπωνυμο', 'ονοματεπώνυμο', 'ονομα', 'όνομα',
        'επωνυμο', 'επώνυμο', 'διευθυνση', 'διεύθυνση',
        'ταυτοτητα', 'ταυτότητα', 'tautotita',
    ],
    # Definite QIs — high confidence
    'definite_qis_gr': {
        # Demographic
        'ηλικια': 1.00, 'ηλικία': 1.00,
        'φυλο': 1.00, 'φύλο': 1.00,
        'γενος': 0.95, 'γένος': 0.95,
        # Geographic hierarchy
        'νομος': 0.95, 'νομός': 0.95,
        'νομαρχια': 0.95, 'νομαρχία': 0.95,
        'δημος': 0.95, 'δήμος': 0.95,
        'περιφερεια': 0.90, 'περιφέρεια': 0.90,
        'κοινοτητα': 0.90, 'κοινότητα': 0.90,
        'πολη': 0.95, 'πόλη': 0.95,
        # Postal
        'ταχυδρομικος': 1.00, 'ταχυδρομικός': 1.00,
        # Education / employment / marital / ethnicity
        'εκπαιδευση': 0.95, 'εκπαίδευση': 0.95,
        'επαγγελμα': 0.95, 'επάγγελμα': 0.95,
        'οικογενειακη': 0.95, 'οικογενειακή': 0.95,
        'υπηκοοτητα': 0.90, 'υπηκοότητα': 0.90,
        'εθνικοτητα': 0.90, 'εθνικότητα': 0.90,
        'θρησκευμα': 0.80, 'θρήσκευμα': 0.80,
        'ιθαγενεια': 0.90, 'ιθαγένεια': 0.90,
    },
    # Probable QIs — medium confidence
    'probable_qis_gr': {
        # Temporal
        'ετος': 0.70, 'έτος': 0.70,
        'μηνας': 0.65, 'μήνας': 0.65,
        'ημερομηνια': 0.60, 'ημερομηνία': 0.60,
        # Administrative
        'τμημα': 0.60, 'τμήμα': 0.60,
        'βαθμιδα': 0.55, 'βαθμίδα': 0.55,
        'κλαδος': 0.60, 'κλάδος': 0.60,
        'ασφαλιση': 0.55, 'ασφάλιση': 0.55,
        # Property / real estate
        'οροφος': 0.60, 'όροφος': 0.60,
        'χρηση': 0.55, 'χρήση': 0.55,
        'ειδος': 0.55, 'είδος': 0.55,
        'προσοψη': 0.50, 'πρόσοψη': 0.50,
        'ζωνη': 0.55, 'ζώνη': 0.55,
        # Geographic sub-levels
        'διαμερισμα': 0.60, 'διαμέρισμα': 0.60,
        'περιοχη': 0.65, 'περιοχή': 0.65,
        'τοπος': 0.60, 'τόπος': 0.60,
    },
}

# Column type keywords for detection
COLUMN_TYPE_KEYWORDS: Dict[str, List[str]] = {
    'identifier': [
        'id', 'uuid', 'guid', 'key', 'number', 'num', 'no', 'code',
        'ssn', 'ein', 'tin', 'account', 'license', 'permit',
        'index', 'idx', 'pk', 'fk', 'serial',
    ],
    'binary': [
        'flag', 'indicator', 'is_', 'has_', 'was_', 'active',
        'enabled', 'disabled', 'yes', 'no', 'true', 'false',
        'valid', 'invalid', 'status',
    ],
    'sensitive': [
        'disease', 'diagnosis', 'condition', 'symptom', 'treatment',
        'medication', 'drug', 'prescription', 'health', 'medical',
        'income', 'salary', 'wage', 'earnings', 'payment', 'debt',
        'credit', 'score', 'rating', 'criminal', 'arrest', 'conviction',
        'orientation', 'preference', 'political', 'religion', 'belief',
        'pregnancy', 'fertility', 'disability', 'hiv', 'aids', 'mental',
        'psychiatric', 'addiction', 'substance', 'alcohol', 'abuse',
        'ssn', 'social_security', 'passport', 'license', 'account',
        'password', 'pin', 'secret', 'token', 'key', 'credential',
        'biometric', 'fingerprint', 'dna', 'genetic',
    ],
    'date': [
        'date', 'time', 'datetime', 'timestamp', 'created', 'updated',
        'modified', 'at', 'on', 'when', 'year', 'month', 'day',
        'birth', 'dob', 'start', 'end', 'begin', 'finish',
        'ημερομηνία', 'ημερομηνια', 'ετος', 'έτος', 'μήνας', 'μηνας',
    ],
}

# =============================================================================
# SENSITIVE VALUE KEYWORDS  (for auto-classify: analytical target columns)
# =============================================================================
# Clean keyword list: ONLY analytical/value columns the user likely wants to
# preserve during SDC.  Does NOT include identifiers (SSN, passport, account)
# or credentials (password, token) – those belong to DIRECT_IDENTIFIER_KEYWORDS.

SENSITIVE_VALUE_KEYWORDS: Dict[str, float] = {
    # ── Medical / health outcomes ──
    'disease': 0.90, 'diagnosis': 0.90, 'condition': 0.85,
    'symptom': 0.85, 'treatment': 0.80, 'medication': 0.80,
    'drug': 0.75, 'prescription': 0.80, 'health': 0.70,
    'medical': 0.70, 'clinical': 0.75, 'prognosis': 0.85,
    'mortality': 0.90, 'morbidity': 0.90, 'complication': 0.80,
    'disability': 0.80, 'mental': 0.75, 'psychiatric': 0.80,
    # ── Financial outcomes ──
    'income': 0.90, 'salary': 0.90, 'wage': 0.85,
    'earnings': 0.85, 'payment': 0.70, 'debt': 0.80,
    'revenue': 0.75, 'profit': 0.75, 'loss': 0.70,
    'expense': 0.70, 'cost': 0.65, 'price': 0.60,
    'tax': 0.75, 'fee': 0.60, 'charge': 0.60,
    'balance': 0.70, 'amount': 0.60, 'total': 0.55,
    'rent': 0.70, 'loan': 0.75, 'subsidy': 0.70,
    'allowance': 0.70, 'benefit': 0.65, 'pension': 0.75,
    # ── Scores and ratings ──
    'score': 0.65, 'rating': 0.60, 'grade': 0.60,
    'rank': 0.55, 'percentile': 0.65,
    # ── Measurements / outcomes ──
    'result': 0.60, 'outcome': 0.70, 'value': 0.45,
    'measure': 0.55, 'count': 0.45, 'frequency': 0.50,
    'quantity': 0.60, 'executed': 0.55,
    'duration': 0.55, 'length': 0.45, 'distance': 0.40,
    'area': 0.50, 'surface': 0.50, 'volume': 0.50,
    'weight': 0.50, 'height': 0.50, 'bmi': 0.70,
    # ── Behavioral / preference ──
    'preference': 0.60, 'satisfaction': 0.65, 'opinion': 0.60,
    'response': 0.50, 'answer': 0.45,
}

SENSITIVE_VALUE_KEYWORDS_GR: Dict[str, float] = {
    # ── Greek medical / health ──
    'νοσημα': 0.90, 'νόσημα': 0.90, 'διαγνωση': 0.90, 'διάγνωση': 0.90,
    'θεραπεια': 0.85, 'θεραπεία': 0.85, 'φαρμακο': 0.80, 'φάρμακο': 0.80,
    'συμπτωμα': 0.85, 'σύμπτωμα': 0.85, 'υγεια': 0.70, 'υγεία': 0.70,
    'αναπηρια': 0.80, 'αναπηρία': 0.80,
    # ── Greek financial ──
    'εισοδημα': 0.90, 'εισόδημα': 0.90, 'μισθος': 0.90, 'μισθός': 0.90,
    'αμοιβη': 0.85, 'αμοιβή': 0.85, 'δαπανη': 0.75, 'δαπάνη': 0.75,
    'κοστος': 0.65, 'κόστος': 0.65, 'τιμη': 0.60, 'τιμή': 0.60,
    'ποσο': 0.65, 'ποσό': 0.65, 'συνολο': 0.55, 'σύνολο': 0.55,
    'εσοδα': 0.75, 'έσοδα': 0.75, 'κερδος': 0.75, 'κέρδος': 0.75,
    'φορος': 0.75, 'φόρος': 0.75, 'ενοικιο': 0.70, 'ενοίκιο': 0.70,
    'δανειο': 0.75, 'δάνειο': 0.75, 'επιδομα': 0.70, 'επίδομα': 0.70,
    'συνταξη': 0.75, 'σύνταξη': 0.75,
    # ── Greek measurement / outcome ──
    'αποτελεσμα': 0.65, 'αποτέλεσμα': 0.65,
    'βαθμος': 0.60, 'βαθμός': 0.60,
    'επιφανεια': 0.50, 'επιφάνεια': 0.50,
    'εμβαδον': 0.50, 'εμβαδόν': 0.50,
    'ογκος': 0.50, 'όγκος': 0.50,
    'βαρος': 0.50, 'βάρος': 0.50,
    'υψος': 0.50, 'ύψος': 0.50,
    'διαρκεια': 0.55, 'διάρκεια': 0.55,
}

# Administrative / classifier keywords (negative signal for sensitive detection)
ADMIN_KEYWORDS: List[str] = [
    'code', 'type', 'category', 'status', 'group', 'class', 'flag',
    'level', 'kind', 'mode', 'phase', 'stage', 'tier', 'section',
    'κωδικος', 'κωδικός', 'τυπος', 'τύπος', 'κατηγορια', 'κατηγορία',
    'κατασταση', 'κατάσταση', 'ομαδα', 'ομάδα',
]

# =============================================================================
# CONTEXT-BASED PROTECTION THRESHOLDS
# =============================================================================

PROTECTION_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    'public_release': {
        'description': 'Data for unrestricted public access',
        'k_min': 10,
        'reid_95_max': 0.01,
        'reid_99_max': 0.03,
        'uniqueness_max': 0.01,
        'info_loss_max': 0.15,
        'suppression_max': 0.10,
        'l_min': 3,
        't_max': 0.15,
        'priority': 'privacy',
    },
    'scientific_use': {
        'description': 'Data for approved research purposes',
        'k_min': 5,
        'reid_95_max': 0.05,
        'reid_99_max': 0.10,
        'uniqueness_max': 0.05,
        'info_loss_max': 0.10,
        'suppression_max': 0.15,
        'l_min': 2,
        't_max': 0.25,
        'priority': 'utility',
    },
    'secure_environment': {
        'description': 'Data for internal organizational use (secure environment)',
        'k_min': 3,
        'reid_95_max': 0.10,
        'reid_99_max': 0.20,
        'uniqueness_max': 0.10,
        'info_loss_max': 0.08,
        'suppression_max': 0.20,
        'l_min': 2,
        't_max': 0.30,
        'priority': 'utility',
    },
    'regulatory_compliance': {
        'description': 'Data meeting regulatory requirements (HIPAA, GDPR)',
        'k_min': 5,
        'reid_95_max': 0.03,
        'reid_99_max': 0.05,
        'uniqueness_max': 0.03,
        'info_loss_max': 0.12,
        'suppression_max': 0.12,
        'l_min': 3,
        't_max': 0.20,
        'priority': 'privacy',
    },
    'default': {
        'description': 'Default balanced protection',
        'k_min': 5,
        'reid_95_max': 0.05,
        'reid_99_max': 0.10,
        'uniqueness_max': 0.05,
        'info_loss_max': 0.12,
        'suppression_max': 0.15,
        'l_min': 2,
        't_max': 0.25,
        'priority': 'balanced',
    },
}

# =============================================================================
# RULE REGISTRY - Data Structure Rules
# =============================================================================

DATA_STRUCTURE_RULES: Dict[str, Dict[str, Any]] = {
    # Tabular rules DS1-DS3 removed — app is microdata only.
    # If tabular data is detected, rules engine falls through to default
    # microdata methods (kANON).
    'DS4_Continuous_Only': {
        'trigger': 'n_continuous>0 AND n_categorical==0',
        'method': 'NOISE',
        'parameters': {'magnitude': 0.15},
        'priority': 'RECOMMENDED',
        'confidence': 'HIGH',
        'reason': 'Continuous-only data - noise appropriate',
    },
    'DS4_Continuous_With_Outliers': {
        'trigger': 'n_continuous>0 AND n_categorical==0 AND has_outliers',
        'method': 'NOISE',
        'parameters': {'magnitude': 0.20},
        'priority': 'RECOMMENDED',
        'confidence': 'HIGH',
        'reason': 'Outliers present - higher noise magnitude for perturbation',
    },
    'DS5_Categorical_Only': {
        'trigger': 'n_categorical>0 AND n_continuous==0',
        'method': 'PRAM',
        'parameters': {'p_change': 0.2},
        'priority': 'RECOMMENDED',
        'confidence': 'HIGH',
        'reason': 'Categorical-only data - PRAM appropriate',
    },
}

# =============================================================================
# RULE REGISTRY - ReID Risk Rules
# =============================================================================

REID_RISK_RULES: Dict[str, Dict[str, Any]] = {
    'QR1_Severe_Tail_Risk': {
        'trigger': 'risk_pattern==severe_tail',
        'method': 'LOCSUPR',
        'parameters': {'k': 5},
        'priority': 'REQUIRED',
        'confidence': 'HIGH',
        'reason': 'Few records dominate risk - targeted suppression',
        'alternatives': ['kANON'],
    },
    'QR2_Moderate_Tail_Risk': {
        'trigger': 'risk_pattern==tail AND reid_95>0.30',
        'method': 'LOCSUPR',
        'parameters': {'k': 3},
        'priority': 'REQUIRED',
        'confidence': 'HIGH',
        'reason': 'Tail risk requires targeted suppression',
        'alternatives': ['kANON'],
    },
    'QR3_Uniform_High_Risk': {
        'trigger': 'risk_pattern==uniform_high',
        'method': 'kANON',
        'parameters': {'k': 10, 'strategy': 'generalization'},
        'priority': 'REQUIRED',
        'confidence': 'HIGH',
        'reason': 'Uniform high risk - widespread protection needed',
    },
    'QR4_Widespread_Categorical': {
        'trigger': 'risk_pattern==widespread AND reid_50>0.15 AND n_categorical>n_continuous',
        'method': 'PRAM',
        'parameters': {'p_change': 0.3},
        'priority': 'REQUIRED',
        'confidence': 'HIGH',
        'reason': 'Widespread risk - PRAM perturbs broadly',
        'alternatives': ['kANON'],
    },
    'QR4_Widespread_Continuous': {
        'trigger': 'risk_pattern==widespread AND reid_50>0.15 AND n_continuous>=n_categorical',
        'method': 'NOISE',
        'parameters': {'magnitude': 0.25},
        'priority': 'REQUIRED',
        'confidence': 'HIGH',
        'reason': 'Widespread risk - noise perturbs broadly',
        'alternatives': ['kANON'],
    },
    'QR5_High_95th_Percentile': {
        'trigger': 'reid_95>0.20 AND reid_50<0.10',
        'method': 'kANON',
        'parameters': {'k': 7, 'strategy': 'hybrid'},
        'priority': 'REQUIRED',
        'confidence': 'HIGH',
        'reason': 'High tail risk - k-anonymity with hybrid strategy',
    },
    'QR6_Bimodal_Risk': {
        'trigger': 'risk_pattern==bimodal',
        'method': 'kANON',
        'parameters': {'k': 5, 'strategy': 'hybrid'},
        'priority': 'RECOMMENDED',
        'confidence': 'MEDIUM',
        'reason': 'Bimodal distribution - k-anonymity handles both groups',
    },
    'QR7_Many_High_Risk': {
        'trigger': 'high_risk_rate>0.10',
        'method': 'kANON',
        'parameters': {'k': 7, 'strategy': 'generalization'},
        'priority': 'RECOMMENDED',
        'confidence': 'MEDIUM',
        'reason': 'Many high-risk records - structural protection needed',
    },
    'QR8_Moderate_Risk': {
        'trigger': '0.10<reid_95<=0.20 AND reid_50<0.08',
        'method': 'PRAM',
        'parameters': {'p_change': 0.2},
        'priority': 'RECOMMENDED',
        'confidence': 'MEDIUM',
        'reason': 'Moderate risk - light perturbation sufficient',
    },
}
# =============================================================================
# RULE REGISTRY - Pipeline Rules
# =============================================================================

PIPELINE_RULES: Dict[str, Dict[str, Any]] = {
    'P1_Mixed_Variables_Dual_Risk': {
        'trigger': 'n_continuous>=2 AND n_categorical>=2 AND (high_cardinality_count>=2 OR uniqueness>0.10) AND has_outliers',
        'pipeline': ['NOISE', 'kANON'],
        'priority': 'REQUIRED',
        'confidence': 'HIGH',
        'reason': 'Mixed data with outliers and high cardinality',
    },
    'P2_Dual_Risk_Widespread_Plus_Tail': {
        'trigger': 'reid_50>0.15 AND reid_99>0.70 AND (reid_99-reid_50)>0.40',
        'pipeline': ['PRAM', 'LOCSUPR'],  # or ['NOISE', 'LOCSUPR']
        'priority': 'REQUIRED',
        'confidence': 'HIGH',
        'reason': 'Dual risk pattern - widespread plus extreme tail',
    },
    'P2_Moderate_With_High_Risk_Subgroup': {
        'trigger': '0.10<reid_95<0.25 AND high_risk_rate>0.15',
        'pipeline': ['kANON', 'LOCSUPR'],
        'priority': 'RECOMMENDED',
        'confidence': 'HIGH',
        'reason': 'Moderate overall but significant high-risk subgroup',
    },
    'P3_Multiple_High_Cardinality_QIs': {
        'trigger': 'high_cardinality_count>=3',
        'pipeline': ['kANON', 'LOCSUPR'],
        'priority': 'REQUIRED',
        'confidence': 'HIGH',
        'reason': 'Multiple high-cardinality QIs create vast combination space',
    },
    # P4a_Skewed_Structural and P4b_Skewed_Sensitive_Targeted deleted in
    # Spec 19 Phase 2 — P4a had a latent KeyError crash, P4b's |skew| > 1.5
    # gate was too narrow for any harness dataset.
    'P5_Small_Dataset_Mixed_Risks': {
        'trigger': 'density<5 AND n_records>=200 AND uniqueness>0.15 AND n_continuous>=2 AND n_categorical>=2',
        'pipeline': ['NOISE', 'PRAM'],
        'priority': 'RECOMMENDED',
        'confidence': 'MEDIUM',
        'reason': 'Sparse dataset with mixed types - NOISE for continuous, PRAM for categorical',
    },
    'P6_Outliers_Plus_High_Cardinality': {
        'trigger': 'has_outliers AND high_cardinality_count>=2',
        'pipeline': ['NOISE', 'kANON'],
        'priority': 'RECOMMENDED',
        'confidence': 'HIGH',
        'reason': 'Outliers perturbed with NOISE, structure protected with kANON',
    },
}

# =============================================================================
# DEFAULT METHOD PARAMETERS
# =============================================================================

DEFAULT_METHOD_PARAMETERS: Dict[str, Dict[str, Any]] = {
    'kANON': {
        'k': 5,
        'strategy': 'generalization',
        'max_suppression_rate': 0.1,
    },
    'PRAM': {
        'p_change': 0.2,
        'invariant': None,
    },
    'NOISE': {
        'magnitude': 0.15,
        'distribution': 'gaussian',
    },
    'LOCSUPR': {
        'k': 3,
        'importance': None,
    },
}

# =============================================================================
# PARAMETER TUNING SCHEDULES
# =============================================================================

PARAMETER_TUNING_SCHEDULES: Dict[str, Dict[str, Any]] = {
    'kANON': {
        'parameter': 'k',
        'values': [3, 5, 7, 10, 15, 20, 25, 30],
        'direction': 'ascending',
        'description': 'Increase k for stronger privacy',
    },
    'PRAM': {
        'parameter': 'p_change',
        'values': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
        'direction': 'ascending',
        'description': 'Increase perturbation probability',
    },
    'NOISE': {
        'parameter': 'magnitude',
        'values': [0.05, 0.10, 0.15, 0.20, 0.25],
        'direction': 'ascending',
        'description': 'Increase noise magnitude',
    },
    'LOCSUPR': {
        'parameter': 'k',
        'values': [3, 5, 7, 10, 15, 20],
        'direction': 'ascending',
        'description': 'Increase k for more suppression',
    },
}

# =============================================================================
# METHOD FALLBACK ORDER
# =============================================================================

METHOD_FALLBACK_ORDER: Dict[str, List[str]] = {
    'kANON': ['LOCSUPR', 'PRAM', 'NOISE'],
    'PRAM': ['kANON', 'LOCSUPR', 'NOISE'],
    'NOISE': ['kANON', 'PRAM', 'LOCSUPR'],
    'LOCSUPR': ['kANON', 'PRAM', 'NOISE'],
}

# =============================================================================
# METRIC → ALLOWED METHODS
# =============================================================================
# Perturbation methods (PRAM, NOISE) cannot guarantee equivalence-class
# properties, so they are blocked when the user targets k-anonymity or
# uniqueness.  ReID (REID95) is universal.  l-diversity allows PRAM
# (perturbation of sensitive values increases diversity) but blocks NOISE.
#
# GENERALIZE / GENERALIZE_FIRST are non-perturbative recoding methods
# (merge categories).  They preserve frequency counts and are compatible
# with every metric.  Missing from this list caused GEO1, RC4, and QR0
# to be silently rejected by _all_allowed() / _is_allowed().
# Fix 0 (pre-Spec 16): added 2026-04-20.

METRIC_ALLOWED_METHODS: Dict[str, List[str]] = {
    'reid95': ['kANON', 'LOCSUPR', 'PRAM', 'NOISE', 'GENERALIZE', 'GENERALIZE_FIRST'],
    'k_anonymity': ['kANON', 'LOCSUPR', 'GENERALIZE', 'GENERALIZE_FIRST'],
    'uniqueness': ['kANON', 'LOCSUPR', 'GENERALIZE', 'GENERALIZE_FIRST'],
    'l_diversity': ['kANON', 'LOCSUPR', 'PRAM', 'GENERALIZE', 'GENERALIZE_FIRST'],
}


# =============================================================================
# PERTURBATIVE CHALLENGE (post-structural-success retry with PRAM)
# =============================================================================

PERTURBATIVE_CHALLENGE: Dict[str, Any] = {
    'enabled': True,
    # Minimum categorical ratio to attempt challenge (at least this fraction
    # of QIs must be categorical for PRAM to be viable)
    'min_cat_ratio': 0.50,
    # Maximum reid_95 at which we bother challenging — above this, PRAM
    # is unlikely to hit target on its own
    'max_reid_95': 0.30,
    # Minimum QI suppression in the structural result that triggers the
    # challenge (no point trying PRAM if structural wasn't destructive)
    'min_structural_suppression': 0.03,
    # Minimum utility improvement (absolute percentage points) required
    # to accept the PRAM result over the structural result
    'min_utility_gain': 0.03,
    # PRAM p_change to use for the challenge, scaled by reid_95
    # p = base_p + reid_95 * scale, capped at max_p
    'base_p': 0.15,
    'scale_p': 0.30,
    'max_p': 0.35,
}


def _normalize_metric_key(risk_metric: str) -> str:
    """Normalize risk metric key."""
    return risk_metric


def is_method_allowed_for_metric(method: str, risk_metric: str) -> bool:
    """Check if *method* is compatible with the user's chosen *risk_metric*."""
    allowed = METRIC_ALLOWED_METHODS.get(_normalize_metric_key(risk_metric))
    if allowed is None:
        return True  # Unknown metric → allow all
    return method in allowed


def filter_methods_for_metric(methods: List[str], risk_metric: str) -> List[str]:
    """Filter a plain list of method-name strings by metric compatibility."""
    allowed = METRIC_ALLOWED_METHODS.get(_normalize_metric_key(risk_metric))
    if allowed is None:
        return methods
    return [m for m in methods if m in allowed]


# =============================================================================
# QI SCORING WEIGHTS
# =============================================================================

QI_SCORING_WEIGHTS: Dict[str, float] = {
    'name_based': 0.20,      # Keyword matching (reduced - names can be misleading)
    'type_based': 0.40,      # Data type and patterns (increased - actual data matters more)
    'uniqueness': 0.40,      # Re-identification contribution (increased - key for risk)
}

QI_CONFIDENCE_TIERS: Dict[str, Dict[str, Any]] = {
    'DEFINITE': {
        'min_score': 0.90,
        'description': 'Clear demographic identifiers',
    },
    'PROBABLE': {
        'min_score': 0.60,
        'description': 'Likely QIs based on characteristics',
    },
    'POSSIBLE': {
        'min_score': 0.30,
        'description': 'Uncertain, may need validation',
    },
    'NON_QI': {
        'min_score': 0.00,
        'description': 'Unlikely to identify individuals',
    },
}

# =============================================================================
# RISK PATTERN THRESHOLDS
# =============================================================================

RISK_PATTERN_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    'uniform_high': {
        'reid_50_min': 0.20,
        'spread_max': 0.10,  # reid_99 - reid_50
    },
    'widespread': {
        'reid_50_min': 0.20,
        'spread_min': 0.10,
    },
    'severe_tail': {
        'reid_50_max': 0.05,
        'reid_95_min': 0.30,
        'reid_99_min': 0.50,
    },
    'tail': {
        'reid_50_max': 0.05,
        'reid_95_min': 0.30,
    },
    'bimodal': {
        'mean_median_diff_min': 0.15,
    },
    'uniform_low': {
        'reid_50_max': 0.05,
        'reid_95_max': 0.30,
    },
    'moderate': {
        'description': 'Default when no other pattern matches',
    },
}

# =============================================================================
# METHOD INFO (for display/documentation)
# =============================================================================

METHOD_INFO: Dict[str, Dict[str, str]] = {
    'kANON': {
        'name': 'k-Anonymity',
        'short': 'Generalize to groups of k',
        'type': 'microdata',
        'preserves': 'Distribution shape',
        'risk': 'Very low for high k',
    },
    'PRAM': {
        'name': 'Post-RAndomization',
        'short': 'Probabilistic category swapping',
        'type': 'microdata',
        'preserves': 'Marginal distributions',
        'risk': 'Low',
    },
    'NOISE': {
        'name': 'Noise Addition',
        'short': 'Add random noise to values',
        'type': 'microdata',
        'preserves': 'Means (with care)',
        'risk': 'Depends on magnitude',
    },
    'LOCSUPR': {
        'name': 'Local Suppression',
        'short': 'Suppress specific cells for k-anon',
        'type': 'microdata',
        'preserves': 'Most values',
        'risk': 'Very low',
    },
}


def get_method_defaults(method: str) -> Dict[str, Any]:
    """Get default parameters for a method."""
    # Try exact match first, then uppercase
    if method in DEFAULT_METHOD_PARAMETERS:
        return DEFAULT_METHOD_PARAMETERS[method].copy()
    return DEFAULT_METHOD_PARAMETERS.get(method.upper(), {}).copy()


def get_tuning_schedule(method: str) -> Dict[str, Any]:
    """Get parameter tuning schedule for a method."""
    # Try exact match first, then uppercase
    if method in PARAMETER_TUNING_SCHEDULES:
        return PARAMETER_TUNING_SCHEDULES[method].copy()
    return PARAMETER_TUNING_SCHEDULES.get(method.upper(), {}).copy()


def get_method_fallbacks(method: str) -> List[str]:
    """Get fallback methods for a given method."""
    # Try exact match first, then uppercase
    if method in METHOD_FALLBACK_ORDER:
        return METHOD_FALLBACK_ORDER[method].copy()
    return METHOD_FALLBACK_ORDER.get(method.upper(), []).copy()


def get_protection_thresholds(context: str = 'default') -> Dict[str, Any]:
    """Get protection thresholds for a given context."""
    return PROTECTION_THRESHOLDS.get(context, PROTECTION_THRESHOLDS['default']).copy()


# =============================================================================
# CONTEXT-TO-TIER MAPPING (shared across all views)
# =============================================================================

CONTEXT_TO_TIER: Dict[str, str] = {
    'public_release': 'PUBLIC',
    'scientific_use': 'SCIENTIFIC',
    'secure_environment': 'SECURE',
    'regulatory_compliance': 'PUBLIC',
    'default': 'SCIENTIFIC',
}


def get_access_tier(context: str) -> str:
    """Map a protection context string to an access tier."""
    return CONTEXT_TO_TIER.get(context, CONTEXT_TO_TIER['default'])


def get_context_targets(context: str, risk_metric: str = 'reid95') -> Dict[str, Any]:
    """Get all targets for a protection context in one call.

    Parameters
    ----------
    context : str
        Protection context key (e.g. 'scientific_use').
    risk_metric : str
        Active risk metric: 'reid95', 'k_anonymity', or 'uniqueness'.

    Returns dict with: reid_target, utility_floor, k_min, uniqueness_max,
    risk_target (metric-specific raw target), risk_target_normalized,
    access_tier, suppression_max, description.
    """
    from sdc_engine.sdc.metrics.risk_metric import (
        RiskMetricType, normalize_target,
    )
    thresholds = get_protection_thresholds(context)

    reid_target = thresholds.get('reid_95_max', 0.05)
    k_min = thresholds.get('k_min', 5)
    uniqueness_max = thresholds.get('uniqueness_max', 0.05)

    # Map string to enum
    _map = {
        'reid95': RiskMetricType.REID95,
        'k_anonymity': RiskMetricType.K_ANONYMITY,
        'uniqueness': RiskMetricType.UNIQUENESS,
    }
    mt = _map.get(risk_metric, RiskMetricType.REID95)

    # Metric-specific primary target (raw units)
    if mt == RiskMetricType.REID95:
        risk_target = reid_target
    elif mt == RiskMetricType.K_ANONYMITY:
        risk_target = k_min
    else:
        risk_target = uniqueness_max

    return {
        'reid_target': reid_target,
        'utility_floor': round(1.0 - thresholds.get('info_loss_max', 0.12), 2),
        'k_min': k_min,
        'uniqueness_max': uniqueness_max,
        'suppression_max': thresholds.get('suppression_max', 0.05),
        'l_target': thresholds.get('l_min'),      # None = disabled
        't_target': thresholds.get('t_max'),       # None = disabled
        'access_tier': get_access_tier(context),
        'description': thresholds.get('description', context),
        'risk_metric': risk_metric,
        'risk_target': risk_target,
        'risk_target_normalized': normalize_target(mt, risk_target),
    }


# =============================================================================
# GENERALIZE TIER PROGRESSION
# =============================================================================

GENERALIZE_TIERS = [
    {'name': 'light', 'max_categories': 15, 'label': 'Light'},
    {'name': 'moderate', 'max_categories': 10, 'label': 'Moderate'},
    {'name': 'aggressive', 'max_categories': 5, 'label': 'Aggressive'},
    {'name': 'very_aggressive', 'max_categories': 3, 'label': 'Very Aggressive'},
]


# =============================================================================
# DEVELOPMENT WORKFLOW SETTINGS
# =============================================================================

DEVELOPMENT_WORKFLOW: Dict[str, Any] = {
    # Documentation update triggers
    'auto_update_docs': True,
    'docs_update_on': [
        'rule_change',           # When method selection rules are modified
        'method_parameter_change',  # When default parameters change
        'new_method',            # When a new SDC method is added
        'api_change',            # When public API functions change
        'threshold_change',      # When protection thresholds change
    ],

    # Documentation files to update
    'docs_to_update': {
        'rule_change': [
            'docs/Method_Selection_Guide.md',
            'docs/API_Reference.md',
        ],
        'method_parameter_change': [
            'docs/User_Guide.md',
            'docs/API_Reference.md',
        ],
        'new_method': [
            'docs/User_Guide.md',
            'docs/API_Reference.md',
            'docs/Method_Selection_Guide.md',
        ],
        'api_change': [
            'docs/API_Reference.md',
            'docs/User_Guide.md',
        ],
        'threshold_change': [
            'docs/Method_Selection_Guide.md',
        ],
    },

    # Git commit settings
    'auto_git_commit': True,
    'commit_on': [
        'major_rule_change',     # Significant rule logic changes
        'new_feature',           # New feature implementation
        'bug_fix',               # Bug fixes
        'performance_improvement',  # Performance optimizations
        'documentation_update',  # Major doc updates
    ],

    # Commit message templates
    'commit_templates': {
        'major_rule_change': 'fix: Update method selection rules for {description}',
        'new_feature': 'feat: Add {description}',
        'bug_fix': 'fix: {description}',
        'performance_improvement': 'perf: {description}',
        'documentation_update': 'docs: Update {description}',
    },

    # After major updates checklist
    'major_update_checklist': [
        'Update relevant documentation files',
        'Run comprehensive tests (tests/test_comprehensive_workflow.py)',
        'Verify ReID reduction for affected methods',
        'Check utility preservation metrics',
        'Create git commit with descriptive message',
        'Update API_Reference.md if public API changed',
        'Update Method_Selection_Guide.md if rules changed',
    ],
}


# Major updates that should trigger documentation and git commit
MAJOR_UPDATE_TYPES: Dict[str, Dict[str, Any]] = {
    'rule_logic_change': {
        'description': 'Changes to method selection rules (QR1-QR8, etc.)',
        'requires_doc_update': True,
        'requires_test': True,
        'docs': ['Method_Selection_Guide.md', 'API_Reference.md'],
        'test_file': 'tests/test_comprehensive_workflow.py',
    },
    'method_behavior_change': {
        'description': 'Changes to how SDC methods work',
        'requires_doc_update': True,
        'requires_test': True,
        'docs': ['User_Guide.md', 'API_Reference.md'],
        'test_file': 'tests/test_methods.py',
    },
    'new_method_added': {
        'description': 'New SDC method implementation',
        'requires_doc_update': True,
        'requires_test': True,
        'docs': ['User_Guide.md', 'API_Reference.md', 'Method_Selection_Guide.md'],
        'test_file': 'tests/test_methods.py',
    },
    'r_integration_change': {
        'description': 'Changes to R/Python implementation defaults',
        'requires_doc_update': True,
        'requires_test': True,
        'docs': ['User_Guide.md', 'Rmeth.md'],
        'test_file': 'tests/test_r_integration.py',
    },
    'detection_logic_change': {
        'description': 'Changes to QI detection or direct identifier detection',
        'requires_doc_update': True,
        'requires_test': True,
        'docs': ['QI_Detection_Guide.md', 'API_Reference.md'],
        'test_file': 'tests/test_detection.py',
    },
}


# =============================================================================
# HIERARCHY BUILDER DEFAULTS (ARX-inspired)
# =============================================================================

HIERARCHY_DEFAULTS: Dict[str, Any] = {
    'numeric_n_levels': 4,
    'date_granularities': ['day', 'month', 'quarter', 'year'],
    'masking_char': '*',
    'categorical_n_levels': 3,
    'categorical_min_frequency': 0.01,
    'auto_build': True,  # auto-build hierarchies when none provided
}

# =============================================================================
# FEATURE PIPELINE PERFORMANCE GUARDS
# =============================================================================

VAR_PRIORITY_COMPUTATION: Dict[str, Any] = {
    'enabled': True,
    # Skip backward elimination for datasets larger than this
    'max_n_records': 10_000,
    # Skip if QI count exceeds this (combinatorial growth)
    'max_n_qis': 8,
    # Timeout for the computation — bail gracefully if it exceeds
    'timeout_seconds': 10.0,
}
