"""
Decode mappings for UCI #697 categorical columns.
Source: UCI dataset documentation (Realinho et al., 2022).
"""

MARITAL_STATUS = {
    1: 'Single', 2: 'Married', 3: 'Widower',
    4: 'Divorced', 5: 'Facto Union', 6: 'Legally Separated'
}

APPLICATION_MODE = {
    1: '1st phase - general contingent',
    2: 'Ordinance No. 612/93',
    5: '1st phase - special contingent (Azores)',
    7: 'Holders of other higher courses',
    10: 'Ordinance No. 854-B/99',
    15: 'International student (bachelor)',
    16: '1st phase - special contingent (Madeira)',
    17: '2nd phase - general contingent',
    18: '3rd phase - general contingent',
    26: 'Ordinance No. 533-A/99, item b2',
    27: 'Ordinance No. 533-A/99, item b3',
    39: 'Over 23 years old',
    42: 'Transfer',
    43: 'Change of course',
    44: 'Technological specialization diploma',
    51: 'Change of institution/course',
    53: 'Short cycle diploma',
    57: 'Change of institution/course (International)'
}

COURSE = {
    33: 'Biofuel Production Technologies',
    171: 'Animation and Multimedia Design',
    8014: 'Social Service (evening attendance)',
    9003: 'Agronomy',
    9070: 'Communication Design',
    9085: 'Veterinary Nursing',
    9119: 'Informatics Engineering',
    9130: 'Equinculture',
    9147: 'Management',
    9238: 'Social Service',
    9254: 'Tourism',
    9500: 'Nursing',
    9556: 'Oral Hygiene',
    9670: 'Advertising and Marketing Management',
    9773: 'Journalism and Communication',
    9853: 'Basic Education',
    9991: 'Management (evening attendance)'
}

PREVIOUS_QUALIFICATION = {
    1: 'Secondary education',
    2: "Higher education - bachelor's degree",
    3: 'Higher education - degree',
    4: "Higher education - master's",
    5: 'Higher education - doctorate',
    6: 'Frequency of higher education',
    9: '12th year of schooling - not completed',
    10: '11th year of schooling - not completed',
    12: 'Other - 11th year of schooling',
    14: '10th year of schooling',
    15: '10th year of schooling - not completed',
    19: 'Basic education 3rd cycle (9th/10th/11th year)',
    38: 'Basic education 2nd cycle (6th/7th/8th year)',
    39: 'Technological specialization course',
    40: 'Higher education - degree (1st cycle)',
    42: 'Professional higher technical course',
    43: 'Higher education - master (2nd cycle)'
}

NACIONALITY = {
    1: 'Portuguese', 2: 'German', 6: 'Spanish', 11: 'Italian',
    13: 'Dutch', 14: 'English', 17: 'Lithuanian', 21: 'Angolan',
    22: 'Cape Verdean', 24: 'Guinean', 25: 'Mozambican',
    26: 'Santomean', 32: 'Turkish', 41: 'Brazilian',
    62: 'Romanian', 100: 'Moldova', 101: 'Mexican',
    103: 'Ukrainian', 105: 'Russian', 108: 'Cuban', 109: 'Colombian'
}

MOTHERS_QUALIFICATION = PREVIOUS_QUALIFICATION  # Same codes
FATHERS_QUALIFICATION = PREVIOUS_QUALIFICATION  # Same codes

MOTHERS_OCCUPATION = {
    0: 'Student', 1: 'Legislative/Executive/Director',
    2: 'Intellectual/Scientific Specialist', 3: 'Intermediate Level Technician',
    4: 'Administrative Staff', 5: 'Personal Services/Security/Seller',
    6: 'Agriculture/Forestry/Fishery', 7: 'Industry/Construction Worker',
    8: 'Machine Operator/Assembly Worker', 9: 'Unskilled Worker',
    10: 'Armed Forces', 90: 'Other Situation', 99: '(Blank)',
    101: 'Armed Forces Officer', 102: 'Armed Forces Sergeant',
    103: 'Armed Forces Other', 112: 'Administrative Staff',
    114: 'Director (Public/Private)', 121: 'Scientific Specialist',
    122: 'Health Specialist', 123: 'Teacher', 124: 'Other Specialist',
    125: 'ICT Specialist', 131: 'Technician Level 1',
    132: 'Technician Level 2', 134: 'Education Technician',
    141: 'Secretary', 143: 'Data/Accounting Operator',
    144: 'Other Admin Support', 151: 'Personal Service Worker',
    152: 'Seller', 153: 'Personal Care Worker', 154: 'Security',
    161: 'Market-oriented Farmer', 163: 'Subsistence Farmer',
    171: 'Construction Worker', 172: 'Metal Worker',
    173: 'Precision/Artisan Worker', 174: 'Printing Worker',
    175: 'Food Processing Worker', 181: 'Machine Operator',
    182: 'Assembly Worker', 183: 'Driver', 192: 'Cleaning Worker',
    193: 'Unskilled Agriculture', 194: 'Meal Preparation',
    195: 'Street Vendor'
}

FATHERS_OCCUPATION = MOTHERS_OCCUPATION  # Same codes

GENDER = {0: 'Female', 1: 'Male'}

YES_NO = {0: 'No', 1: 'Yes'}

TARGET_MAP = {
    'Dropout': 0,
    'Enrolled': 1,
    'Graduate': 2
}

TARGET_INVERSE = {v: k for k, v in TARGET_MAP.items()}


# Map column names to their decoder dictionaries
COLUMN_DECODERS = {
    'Marital status': MARITAL_STATUS,
    'Application mode': APPLICATION_MODE,
    'Course': COURSE,
    'Previous qualification': PREVIOUS_QUALIFICATION,
    'Nacionality': NACIONALITY,
    "Mother's qualification": MOTHERS_QUALIFICATION,
    "Father's qualification": FATHERS_QUALIFICATION,
    "Mother's occupation": MOTHERS_OCCUPATION,
    "Father's occupation": FATHERS_OCCUPATION,
    'Gender': GENDER,
    'Displaced': YES_NO,
    'Educational special needs': YES_NO,
    'Debtor': YES_NO,
    'Tuition fees up to date': YES_NO,
    'Scholarship holder': YES_NO,
    'International': YES_NO,
    'Daytime/evening attendance': {0: 'Evening', 1: 'Daytime'}
}