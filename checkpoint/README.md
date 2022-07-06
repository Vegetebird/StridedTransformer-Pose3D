## Download pretrained model

The pretrained model can be found in [here](https://drive.google.com/drive/folders/1JszQxruPFqux3UzXcJWKgsB67wPk__dH?usp=sharing), please download it and put in the './checkpoint/pretrained' dictory. 

## Test the model

To test on pretrained model on Human3.6M:

```bash
python main.py --test --refine --reload --refine_reload --previous_dir 'checkpoint/pretrained'
```