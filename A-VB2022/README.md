
# ACII A-VB 2022 Workshop & Competition

Baseline code for the three tracks of A-VB 2022 competition. 

Full details and results can be found in the latest release of the [A-VB White Paper](https://arxiv.org/pdf/2207.03572v1.pdf).

**The High-Dimensional Emotion Task (A-VB High)**

The A-VB High track explores a high-dimensional emotion space for understanding vocal bursts. Participants will be challenged with predicting the intensity of 10 emotions (Awe, Excitement, Amusement, Awkwardness, Fear, Horror, Distress, Triumph, Sadness, and Surprise) associated with each vocal burst as a multi-output regression task. Participants will report the average Concordance Correlation Coefficient (CCC) across all 10 emotions. The baseline for this task is set by our end-to-end approach as **0.5686 CCC**


**The Two-Dimensional Emotion Task (A-VB Two)** 

In the A-VB Two track, we investigate a low-dimensional emotion space that is based on the circumplex model of affect. Participants will predict values of arousal and valence (on a scale from 1=unpleasant/subdued, 5=neutral, 9=pleasant/stimulated) as a regression task. Participants will report the average Concordance Correlation Coefficient (CCC), as well as the Pearson correlation coefficient, across the two dimensions. The baseline for this task is set by our end-to-end approach as **0.5084 CCC**

**The Cross-Cultural Emotion Task (A-VB Culture)** 

In the A-VB Culture track, participants will be challenged with predicting the intensity of 10 emotions associated with each vocal burst as a multi-output regression task, using a model or multiple models that generate predictions specific to each of the four cultures (the U.S., China, Venezuela, or South Africa). Specifically, annotations of each vocal burst will consist of culture-specific ground truth, meaning that the ground truth for each sample will be the average of annotations solely from the country of origin of the sample. Participants will report the average Concordance Correlation Coefficient (CCC), as well as the Pearson correlation coefficient, across all 10 emotions. The baseline for this task is set by our end-to-end approach as **0.4401 CCC**

**The Expressive Burst-Type Task (A-VB Type)**

In the A-VB Type task, participants will be challenged with classifying the type of expressive vocal burst from 8 classes (Gasp, Laugh, Cry, Scream, Grunt, Groan, Pant, Other).  The baseline for this task is set by our end-to-end approach as **0.4172 UAR**




----

**Submitting Predictions**

Each team will have 5 opportunities to submit their predictions to `competitions@hume.ai`

In main.py, the predictions are store automatically under `preds/`.  You should submit a comma separated csv file for each task:

`A-VB-High_<team_name>_<submission_no>.csv`
```
File_ID,Awe,Excitement,Amusement,Awkwardness,Fear,Horror,Distress,Triumph,Sadness,Surprise
[10119],0.62710214,0.49645486,0.49803686,0.23848177,0.08278237,0.07724878,0.23759936,0.09026498,0.21404977,0.51422143
```
`A-VB-Two_<team_name>_<submission_no>.csv`
```
File_ID,Valence,Arousal
[10119],0.7631676,0.76780826
```
`A-VB-Culture_<team_name>_<submission_no>.csv`

```
File_ID,China_Awe,China_Excitement,China_Amusement,China_Awkwardness,China_Fear,China_Horror,China_Distress,China_Triumph,China_Sadness,United States_Awe,United States_Excitement,United States_Amusement,United States_Awkwardness,United States_Fear,United States_Horror,United States_Distress,United States_Triumph,United States_Sadness,South Africa_Awe,South Africa_Excitement,South Africa_Amusement,South Africa_Awkwardness,South Africa_Fear,South Africa_Horror,South Africa_Distress,South Africa_Triumph,SouthAfrica_Sadness,Venezuela_Awe,Venezuela_Excitement,Venezuela_Amusement,Venezuela_Awkwardness,Venezuela_Fear,Venezuela_Horror,Venezuela_Distress,Venezuela_Triumph,Venezuela_Sadness,China_Surprise,United States_Surprise,South Africa_Surprise,Venezuela_Surprise
[10119],0.038061377,0.291698,0.124653086,0.022536684,0.0474591,0.066204384,0.3504715,0.053055067,0.1332495,0.4656132,0.16915916,0.20210153,0.12418566,0.058694307,0.05223605,0.17013761,0.036202025,0.14283837,0.27895436,0.22023943,0.27663407,0.14329503,0.06129282,0.051026355,0.14811155,0.043002,0.12653029,0.47265163,0.07118222,0.093341954,0.04070308,0.051994443,0.049771238,0.09662901,0.03959408,0.18467966,0.24399953,0.27347636,0.23664658,0.2085441
```

`A-VB-Type_<team_name>_<submission_no>.csv`
```
File_ID,Voc_Type
[10119],Gasp
```

More information on competition guidelines can be found at [competitions.hume.ai](https://competitions.hume.ai).

Any questions: [competitions@hume.ai](mailto:competitions@hume.ai)

&copy; 2022 **Creative Commons Attribution-Non Commercial-No Derivatives 4.0 International Licence (CC BY-NC-ND)**

