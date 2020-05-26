
ANTUSD contains the file opinion_word.csv, 
where each row is in the following format:

Word,Score,Pos,Neu,Neg,Non,Not

A total of six fields are provided by ANTUSD for each word: 

1. Score: the CopeOpi numerical sentiment score
2. Pos: the number of positive annotations 
3. Neu: the number of neutral annotations 
4. Neg: the number of negative annotations 
5. Non: non-opinionated annotations
6. Not: not-a-word annotations (which is collected from real online segmented data)


The annotation scheme of all related corpora are listed as follows:

				Granularity	Collected Label		Context
NTUSD				Word		Gold			Independent
NTCIR MOAT			Sentence	All			Dependent
Chinese Opinion Treebank	Sentence	All			Dependent
ACBiMA(Cmorph)			Word		Gold			Independent


Note that the number of annotations highly depends on the frequency of the annotated word
as the annotators will only be able to annotate the word when they did read this word.
Those who wants to utilize the information of the number of annotators may want to consider this issue.
 

