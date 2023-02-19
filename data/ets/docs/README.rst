ETS Corpus of Non-Native Written English
----------------------------------------

A corpus of 12,100 TOEFL essays from 11 languages (Arabic, Chinese, French, German, Hindi, Italian, Japanese, Korean,
Spanish, Telugu, and Turkish) that were sampled as evenly possible from 8 retired TOEFL Independent prompts. The dataset
was created with the goal of providing a common corpus for the task of Native Language Identification.

The index files use the following abbreviations ISO 639-3 codes for languages:

====  ========
Code  Language
====  ========
ARA   Arabic
DEU   German
FRA   French
HIN   Hindi
ITA   Italian
JPN   Japanese
KOR   Korean
SPA   Spanish
TEL   Telugu
TUR   Turkish
ZHO   Chinese
====  ========

Files
^^^^^

``data/text/index.csv``
  An index of all of the essays in the set.

  The fields of the index are:

  - Filename
  - Prompt
  - Native Language
  - Proficiency Level

  Possible proficiency levels are:

  - low
  - medium
  - high
  
  There are also derivatives of this file (``index-training.csv``, ``index-dev.csv``, and ``index-test.csv``) that 
  are the index files from the 2013 Native Language Identification Shared Task for the training, development, and
  test sets respectively.


``data/text/prompts/``
  Directory containing prompt texts that the essays are responses to.


``data/text/responses/``
  Directory containing all of the essays/responses. Each filename is in ``data/text/index.csv``. 
  
  There are two sub-directories: ``original`` and ``tokenized``. ``original`` contains the responses before any
  processing was applied, and ``tokenized`` contains the responses after they have gone through ETS's proprietary
  sentence and word tokenizers.


``docs/RR-13-24.pdf``
  ETS Research Report 13-24, "TOEFL11: A Corpus of Non-Native English"
  Authors: Daniel Blanchard, Joel Tetreault, Derrick Higgins, Aoife Cahill, and Martin Chodorow
  This report describes the ETS Corpus of Non-Native English in great detail. It uses the older ETS-internal name
  for the corpus, TOEFL11.
  
``tools/evaluate.py``
  The evaluation script from the 2013 NLI Shared Task.
