# MICO_SST

This repository contains the notebooks used for my runner up solution to the SST-2 portion of the [Microsoft Membership Inference Competition (MICO)](https://github.com/microsoft/MICO).

## Competition

This portion of the competition used the Stanford Sentiment Treebank (SST-2) dataset. The organizers provided a set of models trained on different slices of the dataset. For each model, the competitor was presented with a set of samples, half of which were used for training. My task was to determine with high precision which points from the set were used and which were not. To complicate matters, most models were trained with an algorithm to provide differential privacy guarentees (making it difficult to determine if specific points were used). 

The models were categorized into three sets based on the strength of the differential privacy guarentee of the algorithm used to train. The notebooks in this repository correspond to these three categories (inf = infinite budget i.e. no guarentee, hi = high budget i.e. weak guarentee, and lo = low budget i.e. strong guarentee).

## My Approach

The code in my notebooks build off the starter code provided by the organizers. The notebooks are essentially identical, with the only difference being which scenario of models was being used.

### Basic Idea

A general idea in conducting membership-inference attacks is that models tend to assign higher probabilities to correct answers associated with samples they were trained on versus samples they haven't seen. There are two problems with using this approach naively:

1. Out of sample points similar to in sample points will also be assigned high probability.
2. Some points are just more or less challenging to classify, meaning relative probability isn't a high precision statistic for infering members.

To address these challenges, I compared the probabilities assigned by the target model to those assigned by a reference model which I trained on the entire sst-2 dataset minus the challenge points. This way, the reference model was guarenteed to have not seen the challenge points which the target model had trained on, and both models had not seen the challenge points not used for training. Specifically, I scored points using:

```python
score = target_model(point) - reference_model(point) 
```

My hypothesis was that for unseen data, both models would score the points roughly the same (regardless of the inherant difficulty of the point). On the other hand, if the target model was trained on a point, it would likely score the point higher than the reference model.

### Specifics

I trained all models via the training procedures used by the organizers to train the sst2_inf models, regardless of which category the target model was in (future work could be done to see if using the dp training procedures would improve attack performance on dp models). 

I modified the organizers training code to produce the following train function which was called on the subset of the dataset not used as challenge points:

```python
def train(rest_points):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    device = 'cuda'
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    ds = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(
        pd.DataFrame.from_records(rest_points))}).remove_columns("idx")
    ds = preprocess_text(ds, tokenizer, 67)
    model.train()
    
    training_args = TrainingArguments(
        output_dir='/tmp',
        lr_scheduler_type= 'constant',
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy='no',
        dataloader_num_workers=8,
        per_device_train_batch_size=96,
        gradient_accumulation_steps=1)
    
    trainer = Trainer(
        args = training_args,
        train_dataset = ds['train'],
        model = model,
        tokenizer = tokenizer
    )
    
    trainer.train()
    model.eval()
    return model
```

For each model, I segmented the dataset into the challenge_points and rest_points (all the other data not used for the challenge), trained a reference model on the rest_points, and queried both the target model and reference model on the challenge_points. I computed the scores and normalized them with min-max normalization as per the competition instructions. The following code block is representative of this process, and was repeated for each subset of the competition:

```python
dev_path = os.path.join(CHALLENGE, scenario, 'dev')
for m, model_folder in enumerate(tqdm(sorted(os.listdir(dev_path), key=lambda d: int(d.split('_')[1])), desc="model")):
    data_path = os.path.join(dev_path, model_folder)
    challenge_dataset = ChallengeDataset.from_path(data_path, dataset=dataset, len_training=LEN_TRAINING)
    challenge_points = challenge_dataset.get_challenges()
    challenge_dataloader = torch.utils.data.DataLoader(challenge_points, batch_size=10)
    
    rest_points = challenge_dataset.rest
    ref_model = train(rest_points)
    
    with torch.no_grad():
        model = load_model('sst2', data_path).eval().cuda()
        preds = []
        ref_preds = []
        for batch in challenge_dataloader:
            labels = batch['label'].to(torch.device('cuda'))
            tokenizedSequences = tokenizer(batch['sentence'], return_tensors="pt", padding="max_length", max_length=67)
            tokenizedSequences = tokenizedSequences.to(torch.device('cuda'))

            # query model
            output = model(**tokenizedSequences)
            batch_predictions = F.softmax(output.logits, dim=1)[torch.arange(output.logits.shape[0]), labels].cpu().numpy()
            preds.extend(batch_predictions)

            # ref model
            output = ref_model(**tokenizedSequences)
            batch_predictions = F.softmax(output.logits, dim=1)[torch.arange(output.logits.shape[0]), labels].cpu().numpy()
            ref_preds.extend(batch_predictions)
            
    scores = np.array(preds) - np.array(ref_preds)
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    with open(os.path.join(data_path, "prediction.csv"), "w") as f:
        csv.writer(f).writerow(list(scores))
```

## Disclaimer

I had little time to put toward this compeitition, so code quality is definitely low. Uncommented and duplicate code plauge these notebooks. Beware!

## Acknowledgement

Thank you to the MICO competition organizers and Microsoft in general for hosting this challenge. The community appreciates public challenges, especially those with cash prizes to motivate researchers that otherwise might not have the time. I hope the results of the competition advance the field of privacy in machine learning, and I look forward to seeing the approaches used by other competitors.
