import pandas as pd
import json
import os
import random
from typing import List, Dict, Any

# Create directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/evaluation', exist_ok=True)

# Load the pharmaceutical dataset
df = pd.read_csv('data/raw/drugs_dataset.csv')

# Clean data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the dataset for processing."""
    # Handle missing values
    df = df.fillna({
        'generic_name': 'Unknown',
        'drug_classes': 'Not classified',
        'brand_names': 'No brand names listed',
        'activity': 'unknown',
        'rx_otc': 'Unknown',
        'pregnancy_category': 'N',
        'csa': 'N',
        'alcohol': '',
        'rating': 0.0
    })
    
    # Convert rating to float if it's not already
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0.0)
    
    return df

df = clean_data(df)

# Function to create training examples
def create_training_examples(drug_data: pd.DataFrame) -> List[Dict[str, str]]:
    """Create training examples as input-output pairs."""
    examples = []
    
    for _, drug in drug_data.iterrows():
        # Example 1: General information query
        examples.append({
            "input": f"What can you tell me about {drug.generic_name}?",
            "output": f"{drug.generic_name} (Brand names: {drug.brand_names}) is a medication in the {drug.drug_classes} class. " +
                     f"It is available as {'an over-the-counter medication' if drug.rx_otc == 'OTC' else 'a prescription medication'}. " +
                     f"Its FDA pregnancy category is {drug.pregnancy_category}, and it has an average user rating of {drug.rating}/10 based on patient feedback." +
                     (f" Note that this medication interacts with alcohol." if drug.alcohol == 'X' else "")
        })
        
        # Example 2: Safety during pregnancy
        examples.append({
            "input": f"Is {drug.generic_name} safe during pregnancy?",
            "output": f"{drug.generic_name} has a pregnancy category {drug.pregnancy_category}. " +
                     f"{get_pregnancy_category_description(drug.pregnancy_category)} " +
                     f"Always consult with your healthcare provider before taking any medication during pregnancy."
        })
        
        # Example 3: Alcohol interaction
        examples.append({
            "input": f"Can I drink alcohol while taking {drug.generic_name}?",
            "output": f"{'No, you should avoid alcohol while taking ' + drug.generic_name + '. This medication is known to interact with alcohol, which may increase the risk of side effects or reduce the effectiveness of the medication.' if drug.alcohol == 'X' else 'There are no specific alcohol interactions listed for ' + drug.generic_name + ', but it\'s always best to consult with your healthcare provider about alcohol consumption while taking any medication.'}"
        })
        
        # Example 4: General recommendation based on rating
        examples.append({
            "input": f"How effective is {drug.generic_name}?",
            "output": f"{drug.generic_name} has an average user rating of {drug.rating}/10 based on patient feedback. " +
                     f"{get_rating_interpretation(drug.rating)} " +
                     f"Remember that individual responses to medications can vary, and you should follow your healthcare provider's advice."
        })
        
        # Example 5: Drug class information
        examples.append({
            "input": f"What class of medication is {drug.generic_name}?",
            "output": f"{drug.generic_name} belongs to the {drug.drug_classes} class of medications."
        })
        
        # Example 6: Prescription status
        examples.append({
            "input": f"Do I need a prescription for {drug.generic_name}?",
            "output": f"{'Yes, ' + drug.generic_name + ' requires a prescription from a healthcare provider.' if drug.rx_otc == 'Rx' else 'No, ' + drug.generic_name + ' is available over-the-counter without a prescription.' if drug.rx_otc == 'OTC' else drug.generic_name + ' may be available both as a prescription and over-the-counter depending on the specific formulation and dosage.'}"
        })
        
        # Example 7: Controlled substance information
        if drug.csa not in ['N', 'U']:
            examples.append({
                "input": f"Is {drug.generic_name} a controlled substance?",
                "output": f"Yes, {drug.generic_name} is a Schedule {drug.csa} controlled substance. " +
                         f"{get_csa_description(drug.csa)}"
            })
        else:
            examples.append({
                "input": f"Is {drug.generic_name} a controlled substance?",
                "output": f"No, {drug.generic_name} is not classified as a controlled substance under the Controlled Substances Act."
            })
    
    return examples

# Helper functions
def get_pregnancy_category_description(category: str) -> str:
    """Return description for FDA pregnancy category."""
    descriptions = {
        'A': 'Category A means adequate and well-controlled studies have failed to demonstrate a risk to the fetus in the first trimester of pregnancy, and there is no evidence of risk in later trimesters.',
        'B': 'Category B means animal reproduction studies have failed to demonstrate a risk to the fetus and there are no adequate and well-controlled studies in pregnant women.',
        'C': 'Category C means animal reproduction studies have shown an adverse effect on the fetus and there are no adequate and well-controlled studies in humans, but potential benefits may warrant use in pregnant women despite potential risks.',
        'D': 'Category D means there is positive evidence of human fetal risk based on adverse reaction data, but potential benefits may warrant use in pregnant women despite potential risks.',
        'X': 'Category X means studies in animals or humans have demonstrated fetal abnormalities and/or there is positive evidence of human fetal risk, and the risks involved in use in pregnant women clearly outweigh potential benefits.',
        'N': 'The FDA has not classified this drug for use during pregnancy.'
    }
    return descriptions.get(category, 'The pregnancy category information is not available.')

def get_rating_interpretation(rating: float) -> str:
    """Interpret the user rating."""
    if rating >= 8:
        return 'This is considered highly effective by most users.'
    elif rating >= 6:
        return 'This is considered moderately effective by most users.'
    elif rating > 0:
        return 'This has a lower effectiveness rating from users.'
    else:
        return 'There is insufficient user rating data available for this medication.'

def get_csa_description(csa: str) -> str:
    """Return description for CSA schedule."""
    descriptions = {
        '1': 'Schedule 1 substances have a high potential for abuse and no currently accepted medical use in treatment in the United States.',
        '2': 'Schedule 2 substances have a high potential for abuse which may lead to severe psychological or physical dependence.',
        '3': 'Schedule 3 substances have a potential for abuse less than those in schedules 1 and 2, and abuse may lead to moderate or low physical dependence or high psychological dependence.',
        '4': 'Schedule 4 substances have a low potential for abuse relative to those in schedule 3.',
        '5': 'Schedule 5 substances have a low potential for abuse relative to those in schedule 4.'
    }
    return descriptions.get(csa, '')

# Generate examples
all_examples = create_training_examples(df)

# Split into training and evaluation sets (90/10)
random.seed(42)  # For reproducibility
random.shuffle(all_examples)
split_idx = int(0.9 * len(all_examples))
train_examples = all_examples[:split_idx]
eval_examples = all_examples[split_idx:]

# Save to files
with open('data/processed/train.jsonl', 'w') as f:
    for example in train_examples:
        f.write(json.dumps(example) + '\n')

with open('data/evaluation/eval.jsonl', 'w') as f:
    for example in eval_examples:
        f.write(json.dumps(example) + '\n')

print(f"Created {len(train_examples)} training examples and {len(eval_examples)} evaluation examples.")