# MFA: TDNN with Multi-Scale Frequency-Channel Attention for Text-Independent Speaker Verification with Short Utterances

# Note: 
Due to restrictions imposed by our affiliations, we are unable to share the complete system and pre-trained models. We appreciate everyone's support and attention towards this work. As a result, we have decided to share the model script and training configuration in the hope that it will be helpful for the research community. Important: for academic research purposes only.

# Scripts:
```
MFA-TDNN model script: ECAPA_tc_0813.py
Training Configuration script: train_ecapa_tc_0813.yaml
Testing Configuration script: verification_ecapa_tc_0813.yaml
```

# Usage:
```
The system is built based on SpeechBrain (https://github.com/speechbrain/speechbrain)
Copy the model script 'ECAPA_tc_0813.py' to '/speechbrain/speechbrain/lobes/models/'
Copy the training configuration file 'train_ecapa_tc_0813.yaml' to '/speechbrain/recipes/VoxCeleb/SpeakerRec/hparams/'
  (You may need to change the paths, e.g. rir_folder and data_folder)
Copy the testing configuration file 'verification_ecapa_tc_0813.yaml' to '/speechbrain/recipes/VoxCeleb/SpeakerRec/hparams/'
  (You may need to change the paths, e.g. voxceleb_source, data_folder and pretrain_path)

Training: python train_speaker_embeddings.py hparams/train_ecapa_tc_0813.yaml
Testing: python speaker_verification_cosine.py hparams/verification_ecapa_tc_0813.yaml
```

# Cite:
```
@INPROCEEDINGS{9747021,
  author={Liu, Tianchi and Das, Rohan Kumar and Aik Lee, Kong and Li, Haizhou},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={{MFA: TDNN} with Multi-Scale Frequency-Channel Attention for Text-Independent Speaker Verification with Short Utterances}, 
  year={2022},
  volume={},
  number={},
  pages={7517-7521},
  doi={10.1109/ICASSP43922.2022.9747021}}
```
