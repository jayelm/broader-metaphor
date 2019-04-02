# analysis

This is an analysis of randomly sampled examples from the ELMo LAC model
reported in the paper (which are contained in `./elmo_dev_predictions.csv` and,
due to seeding, should be exactly reproducible by rerunning the ELMo model as
described in the project README).

You can view the analysis (including numbers for paper Table 4 and Section 6)
in in `elmo_dev_analysis.html`. The code to produce this is located in
`elmo_dev_analysis.Rmd`. I recommend using RStudio with the `tidyverse` and
`knitr` packages to reproduce this document, which will also let you more
easily examine the data (and my annotations) yourself.
