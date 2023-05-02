# Everything2Text

## Handwriting

#### Create virtual env
```
virtualenv -p python3.6 .venv
```
#### Activate virtual env
```
source .venv/bin/activate
```
#### Install requirements
```
pip install -r requirements.txt 
```

#### Toy dataset
```
Dataset link: https://drive.google.com/drive/folders/1rzSo74lbslGNMbyUmGxmzoaHnIGuQQi3?usp=sharing

Move `handwriting` directory inside `data` directory.
```

#### Configure Neptune
Create project and get api token from neptune.ai
```
export NEPTUNE_API_TOKEN='API_TOKEN'
```

#### Create char and font vocabulary
```
python -m src.data.build_vocab --type char_vocab --data_file data/handwriting/dataset/text.csv --vocab_file data/handwriting/vocab/char.txt

python -m src.data.build_vocab --type font_vocab --data_file data/handwriting/dataset/text.csv --vocab_file data/handwriting/vocab/font.txt
```

#### Train model
```
python -m src.main --task handwriting --data_type full --neptune_project_name <project_name> 

Eg: python -m src.main --task handwriting --data_type full --neptune_project_name 'mahendrathapa/handwriting-10K'
```
Out folder description:
- Out directory: out/handwriting/<run_id>
- Model directory: out/handwriting/<run_id>/model
- Generate result directory: out/handwriting/<run_id>/gen_result

*To continue the model training and tracking log in the same neptune experiment*
```
python -m src.main --task handwriting --data_type full --run_id <run_id> --model_id <model_id> --neptune_project_name <project_name> --neptune_exp_id <exp_id>

Eg: python -m src.main --task handwriting --data_type full --run_id 1610941293 --model_id 46 --neptune_project_name 'mahendrathapa/handwriting-50K' --neptune_exp_id HAN1-23
```


