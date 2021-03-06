Voice Gender
Gender Recognition by Voice and Speech Analysis

This database was created to identify a voice as male or female, based upon acoustic properties of the voice and speech. The dataset consists of 3,168 recorded voice samples, collected from male and female speakers. The voice samples are pre-processed by acoustic analysis in R using the seewave and tuneR packages, with an analyzed frequency range of 0hz-280hz (human vocal range).

The Dataset
The following acoustic properties of each voice are measured and included within the CSV:

meanfreq: mean frequency (in kHz)
sd: standard deviation of frequency
median: median frequency (in kHz)
Q25: first quantile (in kHz)
Q75: third quantile (in kHz)
IQR: interquantile range (in kHz)
skew: skewness (see note in specprop description)
kurt: kurtosis (see note in specprop description)
sp.ent: spectral entropy
sfm: spectral flatness
mode: mode frequency
centroid: frequency centroid (see specprop)
peakf: peak frequency (frequency with highest energy)
meanfun: average of fundamental frequency measured across acoustic signal
minfun: minimum fundamental frequency measured across acoustic signal
maxfun: maximum fundamental frequency measured across acoustic signal
meandom: average of dominant frequency measured across acoustic signal
mindom: minimum of dominant frequency measured across acoustic signal
maxdom: maximum of dominant frequency measured across acoustic signal
dfrange: range of dominant frequency measured across acoustic signal
modindx: modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range
label: male or female
Accuracy
Baseline (always predict male)
50% / 50%

Logistic Regression
97% / 98%

CART
96% / 97%

Random Forest
100% / 98%

SVM
100% / 99%

XGBoost
100% / 99%

Research Questions
An original analysis of the data-set can be found in the following article:

Identifying the Gender of a Voice using Machine Learning

The best model achieves 99% accuracy on the test set. According to a CART model, it appears that looking at the mean fundamental frequency might be enough to accurately classify a voice. However, some male voices use a higher frequency, even though their resonance differs from female voices, and may be incorrectly classified as female. To the human ear, there is apparently more than simple frequency, that determines a voice's gender.

Questions
What other features differ between male and female voices?
Can we find a difference in resonance between male and female voices?
Can we identify falsetto from regular voices? (separate data-set likely needed for this)
Are there other interesting features in the data?
CART Diagram
CART model

Mean fundamental frequency appears to be an indicator of voice gender, with a threshold of 140hz separating male from female classifications.

References
The Harvard-Haskins Database of Regularly-Timed Speech

Telecommunications & Signal Processing Laboratory (TSP) Speech Database at McGill University, Home

VoxForge Speech Corpus, Home

Festvox CMU_ARCTIC Speech Database at Carnegie Mellon University