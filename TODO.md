# In Code
## Code
[] Create a docstring for every class and function
[] Add error handling 
[] Add testing for each Extractor, or other class 

## Model:
[] Add type validation to conversion of the text input 
[] Add fit_partial function
[] Try dslim/distilbert-NER
[] Create a global label map


# Auxiliary
## Documentation
[] Create a directory description by explaining every file
[] Generate docs html (probably sphinx) for the guide on how to introduce more extractors or extend given extractors
[] Generate UML class diagram for the Adapter classes

## Readme
[] Mention link to rest api
[] Mention link to documentation
[] Include image for the UML architecture
[] Include image to processing diagram


## Considerations
[] For evaluation, I use measure micro and macro metrics. Hence, all predictions across documents (flattened) and per document average (each doc metric)
[] Why I choose which models
[] Open challenges
- Ignore single word names

### Performance Considerations
[] Uvicorn only runs with 1 worker because it is intended to be deployed with auto scaling platform
[] Locust model was run on CPU as configuring docker for GPU is more complex


# Model Training
[] Procedure
[] Preprocessing
- Remove things like "'s"
[] Finetuning
- Dropout
- HingeLoss

# Ideas
[] Use embeddings to detect NER by measuring their likelihood
[x] Use finetunig -> https://medium.com/@whyamit101/fine-tuning-bert-for-named-entity-recognition-ner-b42bcf55b51d
[] Fine tune llama3.1 


# Zero Hour
[x] README with install guide, .env variable description, and run guide
[ ] Make 07_train_and_save as script instead of a notebook
[ ] Add pytests
[ ] Create 04_03_finetune
[ ] Reclaim previous py files
[ ] 