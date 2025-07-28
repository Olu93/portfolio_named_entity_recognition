# %% 
from notebook_config import DATASETS_DIR, EXPERIMENTAL_RESULTS_DIR, FILES_DIR, MODEL_CONFIGS
import pandas as pd
import time
from tqdm import tqdm
from port.entity_extractor import MultiEntityExtractor
from sklearn.model_selection import train_test_split

# %%

df = pd.read_csv(DATASETS_DIR / 'full_data_clean.csv').fillna('')
# %%
df.head()


# %%
# Split data into train and test
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
train_df.head()

# %%
def create_multi_entity_extractor(config):
    """
    Factory function to create a MultiEntityExtractor from a configuration dictionary.
    
    Args:
        config (dict): Configuration dictionary with extractor classes and parameters
        
    Returns:
        MultiEntityExtractor: Configured multi-entity extractor
    """
    extractor = MultiEntityExtractor()
    
    # Add extractors for each entity type
    for entity_type in ["persons", "organizations", "locations"]:
        if entity_type in config:
            extractor_class = config[entity_type]["extractor"]
            params = config[entity_type]["params"]
            extractor.add_extractor(entity_type, extractor_class(**params))
    
    return extractor

# %%
# Create predictions folder
predictions_dir = FILES_DIR / 'predictions'
predictions_dir.mkdir(exist_ok=True)

# Test all models and collect results
results = {}

for config in tqdm(MODEL_CONFIGS, desc="Testing models"):
    model_name = config['name']
    model_info = config['extra_info']
    
    start_time = time.time()
    
    try:
        # Create extractor using factory function
        extractor = create_multi_entity_extractor(config)
        
        # Fit and predict
        fit_start = time.time()
        extractor.fit(train_df['text'], train_df[['persons', 'organizations', 'locations']])
        fit_time = time.time() - fit_start
        
        predict_start = time.time()
        predictions = extractor.predict(test_df['text'])
        predict_time = time.time() - predict_start
        
        total_time = time.time() - start_time
        
        # Save predictions to file as DataFrame with true values
        model_size = model_info.get('model', 'N/A')
        safe_model_size = model_size.replace('/', '_').replace('-', '_')
        predictions_file = predictions_dir / f"{model_name}_{safe_model_size}_predictions.csv"
        
        # Create DataFrame with predictions
        predictions_df = pd.DataFrame(predictions)
        
        # Add true values from test set with TRUE_ prefix
        true_values = test_df[['persons', 'organizations', 'locations']].copy()
        true_values.columns = ['TRUE_persons', 'TRUE_organizations', 'TRUE_locations']
        
        # Combine predictions and true values
        combined_df = pd.concat([predictions_df.reset_index(), true_values.reset_index()], axis=1, ignore_index=True)
        combined_df.to_csv(predictions_file, index=False)
        
        # Store results with all metadata
        results[model_name] = {
            'model_name': model_name,
            **model_info,
            'status': 'success',
            'fit_time': fit_time,
            'predict_time': predict_time,
            'total_time': total_time,
            'error': None,
            'stats': extractor.stats,
            'predictions_file': str(predictions_file)
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = str(e)
        
        results[model_name] = {
            'model_name': model_name,
            **model_info,
            'status': 'error',
            'fit_time': 0,
            'predict_time': 0,
            'total_time': total_time,
            'error': error_msg,
            'stats': {},
            'predictions_file': None
        }
    
    results_df = pd.json_normalize(results.values())
    results_df.to_csv(EXPERIMENTAL_RESULTS_DIR / 'model_evaluation_results.csv', index=False)

# %%
# Save results to CSV