# ICML ExVo 2022 Workshop & Competition

Baseline code for the three tracks of ExVo 2022 competition. Full details and results can be found in the lastest release of the [ExVo White Paper](https://www.competitions.hume.ai/s/ICML_ExVo22_Preliminary_04092022.pdf).

**The Multi-task High-Dimensional Emotion, Age & Country Task ([ExVo-MultiTask](https://github.com/HumeAI/competitions/tree/main/ExVo2022/ExVo-MultiTask))**

In ExVo MultiTask, participants will be challenged with predicting the average intensity of each of 10 emotions perceived in vocal bursts, as well as the speaker's Age and native-country, as a multi-task process. For emotion and age, the participants will perform a regression task, and for native-country, a 4-class classification. Participants will report the Concordance Correlation Coefficient (CCC), for the emotion regression task, Mean Absolute Error (MAE) for Age (in years), and Unweighted Average Recall (UAR) for the native-country classification task. The baseline for this track is based on a combined score computed by the harmonic mean between CCC, (inverted) MAE, and UAR.

The Baseline score to beat on the test set for ExVo-MultiTask is: 0.335 S<sub>MTL</sub>

**The Generative Emotional Vocal Burst Task ([ExVo-Generate](https://github.com/HumeAI/competitions/tree/main/ExVo2022/ExVo-Generate))**

In the ExVo Generate task, participants are tasked with applying generative modeling approaches to produce vocal bursts that are associated with 10 distinct emotions. Each team should submit 1000 machine-generated vocalizations (100 for each class) that differentially convey each of the 10 emotions—“awe,” “fear,” etc.—with maximal intensity and fidelity. **Participants can submit samples for either one or all classes**. The ExVo organization team will provide for computing the Fréchet Inception Score. The final evaluation will incorporate human ratings gathered by Hume AI of a random subset of 5/100 samples per targeted emotion. These ratings of the generated vocal bursts will be gathered using the same methodology used to collect the training data, with each vocal burst judged in terms of the perceived intensity of each target emotion. Generated samples will be evaluated based on the Pearsons between normed (0-1) average intensity ratings for the 10 classes and the identity matrix consisting of dummy variables for each class.

The Baseline score to beat for ExVo-Generate is: 0.094 S<sub>GEN</sub>

**The Few-Shot Emotion Recognition task ([ExVo-FewShot](https://github.com/HumeAI/competitions/tree/main/ExVo2022/ExVo-FewShot))**

In the ExVo Few-Shot task, the participants will predict the same 10 emotions as a multi-output regression task, using a model or multiple models. Participants will be provided with at least 2 samples per speaker in all splits (train, validation, test) and will be tasked with performing two-shot personalized emotion recognition. The subject IDs and corresponding emotion labels with two samples per speaker in the test set will be withheld until a week before the deadline for final evaluation of ExVo Few-Shot models on the test data. Participants will report the Concordance Correlation Coefficient (CCC) across all 10 emotion outputs as an evaluation metric.

The Baseline score to beat on the test set for ExVo-FewShot is: 0.444 CCC

More info on competitions guidelines found [competitions.hume.ai](https://competitions.hume.ai).

Any questions: [competitions@hume.ai](mailto:competitions@hume.ai)

&copy; 2022 **Creative Commons Attribution-Non Commercial-No Derivatives 4.0 International Licence (CC BY-NC-ND)**

