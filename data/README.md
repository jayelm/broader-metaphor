# Data

`vuamc.csv` is the main file here, which contains the 2018 NAACL shared task
verbs dataset in a simple `.csv` format with broader contexts extracted from
the BNC.

2018 NAACL shared task files (and instructions for downloading them) are located in `./ets/`.

The VU Amsterdam Metaphor Corpus, `VUAMC.xml`, is obtained from the [Oxford Text Archive](http://ota.ahds.ac.uk/headers/2541.xml) and distributed under a [Creative Commons Attribution-ShareAlike 3.0 Unported License](http://www.vismet.org/metcor/license.html). Available for non-commercial use on condition that the terms of the BNC Licence are observed and that this header is included in its entirety with any copy distributed.

# Preprocessing

To reproduce `vuamc.csv`:

- Download and unzip the XML version of the BNC, obtainable [here](http://ota.ox.ac.uk/desc/2554).
- Run a StanfordCoreNLP server on port 9000, e.g. `java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000`. We use the default `englishPCFG.ser.gz` parser.
- Run `python parse_vuamc.py --bnc_xml_folder YOUR_PATH_TO_BNC_XML` where `YOUR_PATH_TO_BNC_XML` contains the uncompressed BNC XML folders `{A,B,C,...}`. This will take a few hours and produce `vuamc.csv`.
